#!/usr/bin/env python3
"""
Student rollout scoring — 对每条样本用 student 模型采样 num_generations 次,
计算 reward (tIoU) 的 mean / std。reward_std>0 = student 时对时错 = GRPO 有信号。

与 teacher_inference.py 的区别:
- 用原始视频(无 teacher 标注帧)、无 TEACHER_PREFIX
- do_sample=True, temperature=1.0, num_return_sequences=N (复现训练时的采样)
- 输出每条的 reward_mean / reward_std / 8个iou

用法(8 GPU 并行):
  for i in 0..7: CUDA_VISIBLE_DEVICES=$i python student_rollout_scores.py \
      --anno_path <jsonl> --video_dir <timelens_root> --model_path <base> \
      --output_path <out> --chunk 8 --index $i &
"""
import argparse, json, os, re
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "The format should be: 'The event happens in "
    "<start time> - <end time> seconds'."
)

def iou(a, b):
    max0 = max(a[0], b[0]); min0 = min(a[0], b[0])
    max1 = max(a[1], b[1]); min1 = min(a[1], b[1])
    return max(min1 - max0, 0) / (max1 - min0) if (max1 - min0) > 0 else 0.0

def extract_time(p):
    p = p.lower()
    nums = [float(n) for n in re.findall(r"(\d+\.?\d*)", p)]
    nums = nums[:len(nums)//2*2]
    return [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]

def load_annos(anno_path, video_dir=None):
    annos = []
    with open(anno_path) as f:
        for i, line in enumerate(f):
            if line.strip():
                a = json.loads(line); a["_index"] = i
                vp = a.get("video_path", "")
                if video_dir and vp and not os.path.isabs(vp):
                    a["video_path"] = os.path.join(video_dir, vp)
                annos.append(a)
    return annos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno_path", required=True)
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min_tokens", type=int, default=64)
    ap.add_argument("--total_tokens", type=int, default=14336)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0, help="每个shard最多处理多少条(0=全部, 用于快速冒烟)")
    args = ap.parse_args()
    ds = 32

    annos = load_annos(args.anno_path, args.video_dir)
    annos = annos[args.index::args.chunk]
    if args.max_samples > 0:
        annos = annos[:args.max_samples]
    print(f"shard {args.index}/{args.chunk}: {len(annos)} samples")

    is_local = os.path.isdir(args.model_path)
    lk = dict(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
              device_map="auto", trust_remote_code=True)
    if is_local: lk["local_files_only"] = True
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **lk).eval()
    pk = dict(padding_side="left", do_resize=False, trust_remote_code=True)
    if is_local: pk["local_files_only"] = True
    processor = AutoProcessor.from_pretrained(args.model_path, **pk)

    out_file = f"{args.output_path}_{args.index}.jsonl"
    done = set()
    if os.path.exists(out_file):
        for l in open(out_file):
            if l.strip(): done.add(json.loads(l)["_index"])
        print(f"resume: {len(done)} done")
    out_f = open(out_file, "a")

    for k, anno in enumerate(annos):
        idx = anno.get("_index", k)
        if idx in done: continue
        vp = anno.get("video_path", ""); query = anno.get("query", "")
        span = anno.get("span", [])
        if isinstance(span, list) and span and isinstance(span[0], list):
            gs, ge = span[0]
        elif isinstance(span, list) and len(span) == 2:
            gs, ge = span
        else:
            continue
        if not os.path.exists(vp):
            print(f"[SKIP] no video: {vp}"); continue
        try:
            messages = [{"role": "user", "content": [
                {"type": "video", "video": vp,
                 "min_pixels": int(args.min_tokens*ds*ds),
                 "total_pixels": int(args.total_tokens*ds*ds),
                 "fps": float(args.fps)},
                {"type": "text", "text": GROUNDING_PROMPT.format(query)},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, vkw = process_vision_info(
                messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
            if videos is not None:
                vlist, vmeta = zip(*videos); vlist, vmeta = list(vlist), list(vmeta)
            else:
                vlist, vmeta = None, None
            inputs = processor(text=[text], images=images, videos=vlist, video_metadata=vmeta,
                               return_tensors="pt", do_resize=False, **vkw).to("cuda")
            # 循环采样 N 次 (num_return_sequences 对Qwen3-VL视频有split bug, 故逐次生成)
            prompt_len = inputs.input_ids.shape[1]
            answers = []
            with torch.no_grad():
                for _ in range(args.num_generations):
                    out = model.generate(
                        **inputs, do_sample=True, temperature=args.temperature,
                        top_p=1.0, max_new_tokens=args.max_new_tokens)
                    answers.append(
                        processor.batch_decode([out[0][prompt_len:]], skip_special_tokens=True)[0])
            ious = []
            for a in answers:
                ts = extract_time(a)
                ious.append(iou([gs, ge], list(ts[0])) if ts else 0.0)
            rec = {
                "_index": idx, "query": query, "span": [gs, ge],
                "reward_mean": round(float(np.mean(ious)), 4),
                "reward_std": round(float(np.std(ious)), 4),
                "n_correct_ge05": int(sum(1 for x in ious if x >= 0.5)),
                "ious": [round(x, 3) for x in ious],
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (k+1) % 50 == 0:
                out_f.flush()
                print(f"[{k+1}/{len(annos)}] mean={rec['reward_mean']} std={rec['reward_std']}")
        except Exception as e:
            print(f"[ERROR] idx={idx}: {e}")
    out_f.close()
    print(f"Done: {out_file}")

if __name__ == "__main__":
    main()
