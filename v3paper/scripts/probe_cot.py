#!/usr/bin/env python3
"""探针: SFT模型能不能产出 grounding CoT? 同时跑[裸答案]和[先思考再答]两个prompt,
记录两者输出全文+tIoU。实时写入 + 可续跑(按_idx跳过)。

用法(8卡): for i in 0..7: CUDA_VISIBLE_DEVICES=$i python probe_cot.py \
   --model_path <sft> --anno_path <jsonl> --video_dir <root> \
   --output <out> --max_samples 50 --chunk 8 --index $i &
输出 <out>_<index>.jsonl(每行一个样本)。
"""
import argparse, json, os, re
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

PLAIN = ("Please find the visual event described by the sentence '{}', "
         "determining its starting and ending times. The format should be: "
         "'The event happens in <start time> - <end time> seconds'.")

THINK = ("Please find the visual event described by the sentence '{}', "
         "determining its starting and ending times. "
         "First reason step by step about what is happening in the video and when the "
         "described event begins and ends, based only on the visual content you observe. "
         "Then, on the last line, give the final answer in the exact format: "
         "'The event happens in <start time> - <end time> seconds'.")


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    u = max(a[1], b[1]) - min(a[0], b[0])
    return max(hi - lo, 0) / u if u > 0 else 0.0


def extract_answer(text):
    """取答案span: 优先匹配 'happens in X - Y'(最后一处), 否则取文本最后两个数字。"""
    ms = re.findall(r"happens in\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", text.lower())
    if ms:
        s, e = ms[-1]
        return [float(s), float(e)]
    nums = re.findall(r"(\d+\.?\d*)", text)
    if len(nums) >= 2:
        return [float(nums[-2]), float(nums[-1])]
    return None


def build(video, q, tmpl, fps, total_tokens):
    ds = 32
    return [{"role": "user", "content": [
        {"type": "video", "video": video,
         "min_pixels": 64 * ds * ds, "total_pixels": total_tokens * ds * ds, "fps": fps},
        {"type": "text", "text": tmpl.format(q)}]}]


def gen(messages, proc, model, max_new):
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids, vk = process_vision_info(messages, image_patch_size=16,
                                         return_video_kwargs=True, return_video_metadata=True)
    if vids is not None:
        vl, vm = zip(*vids); vl, vm = list(vl), list(vm)
    else:
        vl, vm = None, None
    inp = proc(text=[text], images=imgs, videos=vl, video_metadata=vm,
               return_tensors="pt", do_resize=False, **vk).to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, do_sample=False, max_new_tokens=max_new)
    return proc.batch_decode([out[0][inp.input_ids.shape[1]:]], skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--anno_path", required=True)
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--total_tokens", type=int, default=14336)
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    annos = []
    for i, line in enumerate(open(args.anno_path)):
        if line.strip():
            a = json.loads(line); a["_idx"] = i
            vp = a.get("video_path", "")
            if args.video_dir and vp and not os.path.isabs(vp):
                a["video_path"] = os.path.join(args.video_dir, vp)
            annos.append(a)
    annos = annos[args.index::args.chunk]
    if args.max_samples > 0:
        annos = annos[:args.max_samples]

    outpath = f"{args.output}_{args.index}.jsonl"
    done = set()
    if os.path.exists(outpath):
        for l in open(outpath):
            try: done.add(json.loads(l)["_idx"])
            except Exception: pass
    print(f"[idx{args.index}] 待跑 {len(annos)} 个, 已完成 {len(done)} 个(续跑)", flush=True)

    il = os.path.isdir(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map="auto", trust_remote_code=True, local_files_only=il).eval()
    proc = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=il)

    outf = open(outpath, "a")
    for anno in annos:
        if anno["_idx"] in done:
            continue
        sp = anno["span"]; gt = (sp[0] if isinstance(sp[0], list) else sp)
        gt = [float(gt[0]), float(gt[1])]
        if not os.path.exists(anno["video_path"]):
            continue
        q = anno["query"]
        rec = {"_idx": anno["_idx"], "query": q, "gt": gt}
        try:
            po = gen(build(anno["video_path"], q, PLAIN, args.fps, args.total_tokens), proc, model, 64)
            pa = extract_answer(po)
            rec["plain_out"] = po[:200]
            rec["plain_tiou"] = round(iou(gt, pa), 4) if pa else 0.0
        except Exception as e:
            rec["plain_err"] = str(e)[:120]
        try:
            to = gen(build(anno["video_path"], q, THINK, args.fps, args.total_tokens), proc, model, 512)
            ta = extract_answer(to)
            rec["think_out"] = to[:1500]          # 存全文(截1500字)供人工读CoT质量
            rec["think_len"] = len(to)
            rec["think_tiou"] = round(iou(gt, ta), 4) if ta else 0.0
        except Exception as e:
            rec["think_err"] = str(e)[:120]
        outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        outf.flush()                              # 实时落盘
    outf.close()
    print(f"[idx{args.index}] done", flush=True)


if __name__ == "__main__":
    main()
