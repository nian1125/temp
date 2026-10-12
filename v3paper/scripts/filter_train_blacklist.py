#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练集 jsonl 中剔除黑名单视频(转码失败/损坏), 生成过滤后的 jsonl.

用法:
    python3 scripts/filter_train_blacklist.py \
        --anno   output/2d_sets/train_839_local.jsonl \
        --blacklist output/transcode_failures.txt \
        --out    output/2d_sets/train_local_filtered.jsonl
"""
import os, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default=os.path.join(ROOT, "output/2d_sets/train_839_local.jsonl"))
    ap.add_argument("--blacklist", default=os.path.join(ROOT, "output/transcode_failures.txt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output/2d_sets/train_local_filtered.jsonl"))
    args = ap.parse_args()

    black = set()
    if os.path.isfile(args.blacklist):
        with open(args.blacklist) as f:
            for line in f:
                bn = line.strip()
                if bn:
                    black.add(bn)
    print(f"黑名单视频数: {len(black)}")

    kept, dropped = 0, 0
    with open(args.anno) as f, open(args.out, "w") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            bn = str(obj.get("video_path", "")).split("/")[-1]
            if bn in black:
                dropped += 1
                continue
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"保留 {kept} 条, 剔除 {dropped} 条")
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
