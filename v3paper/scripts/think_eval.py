#!/usr/bin/env python3
"""thinking模型评测: plain视频 + 思考prompt, 末行取答案算mIoU(Charades-Bench)。
实时写+可续。用法(8卡): CUDA_VISIBLE_DEVICES=$i python think_eval.py --model_path <ckpt> ... --index $i
"""
import argparse, json, os, re, sys
import torch
sys.path.insert(0, "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3")
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

THINK = ("Please find the visual event described by the sentence '{}', "
         "determining its starting and ending times. "
         "First reason step by step about what happens in the video and when the "
         "described event begins and ends, based only on the visual content you observe. "
         "Then, on the last line, give the final answer in the exact format: "
         "'The event happens in <start time> - <end time> seconds'.")

ANCHOR = ("Please find the visual event described by the sentence '{}', "
          "determining its starting and ending times. "
          "Reason step by step. After each reasoning step, output your current best estimate "
          "on its own line, exactly as 'estimate: <start> - <end>'. Start from a rough estimate "
          "and progressively refine it as you examine the visual content more carefully. "
          "Finally, on the last line, give the answer exactly as: "
          "'The event happens in <start time> - <end time> seconds'.")

SOFTRANGE = ("Please find the visual event described by the sentence '{}', "
             "determining its starting and ending times. "
             "Give the answer in the exact format: 'The event happens in <start> - <end> seconds.' "
             "Then, on the next line, state how many seconds your start/end might be off, "
             "exactly as: 'uncertainty: <u> seconds'. Be calibrated: small u when confident, "
             "larger when unsure, as tight as possible while still covering the true event.")


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    u = max(a[1], b[1]) - min(a[0], b[0])
    return max(hi - lo, 0) / u if u > 0 else 0.0


def extract_answer(text):
    ms = re.findall(r"happens in\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", text.lower())
    if ms:
        return [float(ms[-1][0]), float(ms[-1][1])]
    nums = re.findall(r"(\d+\.?\d*)", text)
    return [float(nums[-2]), float(nums[-1])] if len(nums) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--anno_path", required=True)
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--total_tokens", type=int, default=14336)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--anchor", action="store_true", help="用anchor渐进prompt(grounded-CoT)")
    ap.add_argument("--softrange", action="store_true", help="用soft边界prompt(自报uncertainty)")
    args = ap.parse_args()
    ds = 32
    PROMPT = ANCHOR if args.anchor else (SOFTRANGE if args.softrange else THINK)

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

    il = os.path.isdir(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map="auto", trust_remote_code=True, local_files_only=il).eval()
    proc = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=il)
    _ps = 16
    for _attr in ("image_processor", "video_processor"):
        _p = getattr(proc, _attr, None); _v = getattr(_p, "patch_size", None) if _p else None
        if _v: _ps = int(_v); break

    outf = open(outpath, "a")
    for anno in annos:
        if anno["_idx"] in done or not os.path.exists(anno["video_path"]):
            continue
        sp = anno["span"]; gt = (sp[0] if isinstance(sp[0], list) else sp)
        gt = [float(gt[0]), float(gt[1])]
        try:
            content = [
                {"type": "video", "video": anno["video_path"],
                 "min_pixels": 64 * ds * ds, "total_pixels": args.total_tokens * ds * ds, "fps": args.fps},
                {"type": "text", "text": PROMPT.format(anno["query"])},
            ]
            msgs = [{"role": "user", "content": content}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids, vk = process_vision_info(msgs, image_patch_size=_ps,
                                                 return_video_kwargs=True, return_video_metadata=True)
            vl = vm = None
            if vids is not None:
                vl, vm = zip(*vids); vl, vm = list(vl), list(vm)
            inp = proc(text=[text], images=imgs, videos=vl, video_metadata=vm,
                       return_tensors="pt", do_resize=False, **vk).to("cuda")
            with torch.no_grad():
                out = model.generate(**inp, do_sample=False, max_new_tokens=512)
            ans = proc.batch_decode([out[0][inp.input_ids.shape[1]:]], skip_special_tokens=True)[0]
            pa = extract_answer(ans)
            t = round(iou(gt, pa), 4) if pa else 0.0
            outf.write(json.dumps({"_idx": anno["_idx"], "tiou": t, "out": ans[:120]}, ensure_ascii=False) + "\n")
        except Exception as e:
            outf.write(json.dumps({"_idx": anno["_idx"], "err": str(e)[:150]}, ensure_ascii=False) + "\n")
        outf.flush()
    outf.close()
    print(f"[idx{args.index}] done", flush=True)


if __name__ == "__main__":
    main()
