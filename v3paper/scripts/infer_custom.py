#!/usr/bin/env python3
"""对自定义 JSONL 数据集做推理。支持 --redblue 模式。
用法:
  # 普通推理
  CUDA_VISIBLE_DEVICES=0 python scripts/infer_custom.py \
    --model_path <path> --anno_path <path> --output <path>
  # 红蓝框推理
  CUDA_VISIBLE_DEVICES=0 python scripts/infer_custom.py \
    --model_path <path> --anno_path <path> --output <path> --redblue
"""
import argparse, hashlib, json, os, re, sys, types
from collections import defaultdict

import nncore
import torch
from nncore.engine import set_random_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMELENS_ROOT = os.environ.get("TIMELENS_ROOT", "/mnt/250T_ceph/zhenjunchen/code/TimeLens")
if TIMELENS_ROOT not in sys.path:
    sys.path.insert(0, TIMELENS_ROOT)

from evaluation.utils import GroundingDataset
from timelens.utils import extract_time


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    u = max(a[1], b[1]) - min(a[0], b[0])
    return max(hi - lo, 0) / u if u > 0 else 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--anno_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--video_dir", default=None)
    p.add_argument("--min_tokens", type=int, default=64)
    p.add_argument("--total_tokens", type=int, default=14336)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--redblue", action="store_true")
    p.add_argument("--redblue_cache", default="/dockerdata/zhenjunchen/cache/redblue_2500")
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_annos(path, video_dir=None):
    annos = []
    for i, line in enumerate(open(path)):
        if not line.strip(): continue
        a = json.loads(line)
        a["_idx"] = i
        # 处理相对路径: video_shards/{ds}/{ds}/{video}.mp4 (多一层嵌套)
        vp = a.get("video_path", "")
        if video_dir and vp and not os.path.isabs(vp):
            parts = vp.split("/", 1)
            if len(parts) == 2:
                a["video_path"] = os.path.join(video_dir, parts[0], parts[0], parts[1])
            else:
                a["video_path"] = os.path.join(video_dir, vp)
        annos.append(a)
    return annos


def apply_redblue_replacements(annos, cache_dir):
    """与 eval_dataloader.py 一致的 redblue 缓存查找逻辑"""
    video_spans = defaultdict(list)
    for a in annos:
        sp = a["span"]
        gt = sp[0] if isinstance(sp[0], (list, tuple)) else sp
        video_spans[a["video_path"]].append((float(gt[0]), float(gt[1])))

    miss = 0
    for a in annos:
        vp = a["video_path"]
        sp = a["span"]
        gt = sp[0] if isinstance(sp[0], (list, tuple)) else sp
        gt = [float(gt[0]), float(gt[1])]
        all_span_list = sorted(video_spans[vp])
        other_key = "-".join([f"{s[0]:.1f}_{s[1]:.1f}" for s in all_span_list])
        key = hashlib.md5(
            f"{vp}_{gt[0]:.2f}_{gt[1]:.2f}_1.5_{other_key}".encode()
        ).hexdigest()[:16]
        cached = os.path.join(cache_dir, f"rb_{key}.mp4")
        if os.path.exists(cached):
            a["video_path"] = cached
        else:
            miss += 1
    if miss:
        print(f"[RedBlue] {miss} cache misses (will use original videos)")
    return annos


if __name__ == "__main__":
    args = parse_args()
    args.seed = set_random_seed(args.seed)

    output_path = f"{args.output}_{args.index}.jsonl"

    annos = load_annos(args.anno_path, args.video_dir)
    annos.sort(key=lambda x: x["duration"], reverse=True)
    annos = annos[args.index::args.chunk]

    if args.redblue:
        annos = apply_redblue_replacements(annos, args.redblue_cache)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        attn_implementation="sdpa", device_map="auto",
        trust_remote_code=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path, padding_side="left",
        do_resize=False, trust_remote_code=True,
    )

    ns = types.SimpleNamespace(
        model_path=args.model_path,
        min_tokens=args.min_tokens, total_tokens=args.total_tokens,
        fps=args.fps, seed=args.seed,
        coarse_box=False, coarse_box_cache="",
        redblue=False, redblue_cache="",
        prompt_file=args.prompt_file,
    )
    dataset = GroundingDataset(annos, processor, ns)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, prefetch_factor=2,
        pin_memory=True, collate_fn=lambda x: x[0],
    )

    outf = open(output_path, "w")
    for data in nncore.ProgressBar(loader):
        inputs = data["inputs"].to("cuda", non_blocking=True)
        anno = data["anno"]

        output_ids = model.generate(
            **inputs, do_sample=False, temperature=None,
            top_p=None, top_k=None, max_new_tokens=512,
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        ans = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

        ts_list = extract_time(ans)
        gt = anno["span"]
        if isinstance(gt[0], (list, tuple)):
            gt = gt[0]
        gt = [float(gt[0]), float(gt[1])]
        if len(ts_list) > 0:
            ts = ts_list[0]
            tiou_val = round(iou(gt, ts), 4)
        else:
            ts = [anno["duration"] + 10, anno["duration"] + 20]
            tiou_val = 0.0

        outf.write(json.dumps({
            "_idx": anno["_idx"],
            "video_path": anno.get("_orig_video_path", anno["video_path"])[:80],
            "query": anno["query"],
            "gt_span": gt,
            "pred_span": [round(ts[0], 1), round(ts[1], 1)],
            "tiou": tiou_val,
            "answer": ans[:200],
        }, ensure_ascii=False) + "\n")
        outf.flush()
    outf.close()
    print(f"[idx{args.index}] done. -> {output_path}")
