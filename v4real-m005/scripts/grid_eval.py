#!/usr/bin/env python3
"""G1 faithful T2SGrid 评测: 模型用 grid 表示输入, 在 Charades-Bench 上算 mIoU。
实时写+可续。用法(8卡): for i: CUDA_VISIBLE_DEVICES=$i python grid_eval.py \
  --model_path <ckpt> --anno_path <bench.jsonl> --video_dir <root> --output <out> --chunk 8 --index $i &
"""
import argparse, json, os, re, sys
import torch
sys.path.insert(0, "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3")
from types import SimpleNamespace
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor
from zwz.data.tvg_opd_dataset import _build_t2sgrid_content

GROUNDING = ("Please find the visual event described by the sentence '{}', "
             "determining its starting and ending times. The format should be: "
             "'The event happens in <start time> - <end time> seconds'.")


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
    ap.add_argument("--min_tokens", type=int, default=64)
    ap.add_argument("--grid_rows", type=int, default=3)
    ap.add_argument("--grid_cols", type=int, default=3)
    ap.add_argument("--grid_overlap", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    da = SimpleNamespace(fps=args.fps, total_tokens=args.total_tokens, min_tokens=args.min_tokens,
                         fps_max_frames=None, grid_rows=args.grid_rows, grid_cols=args.grid_cols,
                         grid_overlap=args.grid_overlap)

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
            content = _build_t2sgrid_content(anno["video_path"], da)
            content = content + [{"type": "text", "text": GROUNDING.format(anno["query"])}]
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
                out = model.generate(**inp, do_sample=False, max_new_tokens=64)
            ans = proc.batch_decode([out[0][inp.input_ids.shape[1]:]], skip_special_tokens=True)[0]
            pa = extract_answer(ans)
            t = round(iou(gt, pa), 4) if pa else 0.0
            outf.write(json.dumps({"_idx": anno["_idx"], "tiou": t, "out": ans[:80]}, ensure_ascii=False) + "\n")
        except Exception as e:
            outf.write(json.dumps({"_idx": anno["_idx"], "err": str(e)[:150]}, ensure_ascii=False) + "\n")
        outf.flush()
    outf.close()
    print(f"[idx{args.index}] done", flush=True)


if __name__ == "__main__":
    main()
