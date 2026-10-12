#!/usr/bin/env python3
"""生成 coarse_box 标注视频，不推理。
遍历 charades-timelens 的每个样本，生成 1.5× 红框标注视频，输出到指定目录。

Usage:
    python scripts/generate_coarse_box_videos.py \
        --bench_path /mnt/250T_ceph/zhenjunchen/data/TimelensBench/charades-timelens.json \
        --video_dir /mnt/250T_ceph/zhenjunchen/data/TimelensBench/video_shards/charades/charades \
        --output_dir /mnt/250T_ceph/zhenjunchen/data/coarse_box_15x_videos \
        --cache_dir /tmp/tvg_teacher_eval_cache \
        --index 0 --chunk 4  # 可选：分片并行
"""

import argparse, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_teacher_strategies import (
    load_timelens_bench,
    create_teacher_video,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_path", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default="/tmp/tvg_teacher_eval_cache")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    annos = load_timelens_bench(args.bench_path, args.video_dir)
    print(f"Total annotations: {len(annos)}")

    # 分片
    annos = annos[args.index::args.chunk]
    print(f"Shard {args.index}/{args.chunk}: {len(annos)} samples")

    mapping = {}  # original_video_id -> annotated_video_path

    for i, anno in enumerate(annos):
        vid = anno["video_id"]
        video_path = anno["video_path"]
        gt_start, gt_end = anno["span"]

        if not os.path.exists(video_path):
            print(f"[SKIP] Video not found: {video_path}")
            continue

        try:
            # 生成 1.5× 窗口红框标注视频
            annotated = create_teacher_video(
                video_path, gt_start, gt_end, args.cache_dir, "coarse_box"
            )
            mapping[vid] = annotated

            if (i + 1) % 50 == 0 or (i + 1) == len(annos):
                print(f"[{i+1}/{len(annos)}] {vid} -> {annotated}")

        except Exception as e:
            print(f"[ERROR] {vid}: {e}")

    # 保存映射
    mapping_file = os.path.join(args.output_dir, f"mapping_shard_{args.index}.json")
    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\nShard {args.index} done. {len(mapping)} videos generated.")
    print(f"Mapping saved to: {mapping_file}")


if __name__ == "__main__":
    main()
