#!/usr/bin/env python3
"""Extract the exact 2500 samples TimeLens uses for GRPO training.

Replicates the sampling logic from TimeLens/training/data/grounding.py
with the same parameters used in run_grpo_qwen3_8b.sh:
  target_size=2500, gaussian_filter_mean=0.05, gaussian_filter_std=0.2,
  fixed_gaussian_sampling=True, seed=42

Usage (on training machine):
  python scripts/extract_timelens_2500.py \
    --anno_path /path/to/gemini_refined_data.jsonl \
    --output_path /path/to/timelens_2500.jsonl
"""
import argparse
import json
import random

import numpy as np


def load_annos(path):
    annos = []
    with open(path) as f:
        for line in f:
            if line.strip():
                annos.append(json.loads(line))
    return annos


def basic_filter(annos, min_video_len=5, max_video_len=500, max_num_words=200):
    filtered = []
    for anno in annos:
        duration = anno.get("duration", 0)
        if duration < min_video_len or duration > max_video_len:
            continue
        num_words = len(anno.get("query", "").split(" "))
        if num_words > max_num_words:
            continue
        span = anno.get("span")
        if span is None:
            continue
        if isinstance(span[0], (int, float)):
            span = [span]
        if not any(0 <= s <= e <= duration for s, e in span):
            continue
        filtered.append(anno)
    return filtered


def gaussian_sample(annos, target_size=2500, seed=42,
                    gaussian_mean=0.05, gaussian_std=0.2):
    duration_ranges = [(i, i + 30) for i in range(0, 240, 30)] + [(240, float("inf"))]
    per_range = int(target_size / len(duration_ranges))

    rng = np.random.default_rng(seed)

    buckets = {dr: [] for dr in duration_ranges}
    kept_indices = []

    for idx, anno in enumerate(annos):
        matched = False
        for dr in buckets:
            lo, hi = dr
            if lo <= anno.get("duration", 0) <= hi:
                buckets[dr].append(idx)
                matched = True
                break
        if not matched:
            kept_indices.append(idx)

    for dr, indices in buckets.items():
        if not indices:
            continue
        num_to_select = min(per_range, len(indices))

        iou_list = np.array(
            [annos[idx].get("iou", 0.0) for idx in indices], dtype=np.float64
        )
        weights = np.exp(-0.5 * ((iou_list - gaussian_mean) / gaussian_std) ** 2)

        # fixed_gaussian_sampling: inverse density correction
        num_bins = 20
        counts, bin_edges = np.histogram(iou_list, bins=num_bins, range=(0, 1))
        bin_indices = np.digitize(iou_list, bins=bin_edges)
        bin_indices = np.clip(bin_indices, 1, num_bins) - 1
        inverse_density = 1.0 / (counts + 1e-6)
        weights *= inverse_density[bin_indices]

        weights = weights / weights.sum()
        selected = rng.choice(indices, size=num_to_select, replace=False, p=weights)
        kept_indices.extend(selected.tolist())

    return [annos[i] for i in sorted(kept_indices)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", required=True,
                        help="Path to gemini_refined_data.jsonl")
    parser.add_argument("--output_path", default="timelens_2500.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--target_size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading annotations from {args.anno_path}")
    raw = load_annos(args.anno_path)
    print(f"  Raw: {len(raw)} samples")

    filtered = basic_filter(raw)
    print(f"  After basic filter: {len(filtered)} samples")

    selected = gaussian_sample(filtered, target_size=args.target_size, seed=args.seed)
    print(f"  After Gaussian sampling: {len(selected)} samples")

    # IoU distribution stats
    ious = [s.get("iou", 0) for s in selected]
    print(f"\n  IoU stats of selected samples:")
    print(f"    mean={np.mean(ious):.4f}, median={np.median(ious):.4f}")
    print(f"    min={np.min(ious):.4f}, max={np.max(ious):.4f}")
    print(f"    IoU<0.1: {sum(1 for x in ious if x < 0.1)} "
          f"({sum(1 for x in ious if x < 0.1)/len(ious)*100:.1f}%)")
    print(f"    IoU<0.3: {sum(1 for x in ious if x < 0.3)} "
          f"({sum(1 for x in ious if x < 0.3)/len(ious)*100:.1f}%)")
    print(f"    IoU>0.5: {sum(1 for x in ious if x > 0.5)} "
          f"({sum(1 for x in ious if x > 0.5)/len(ious)*100:.1f}%)")

    with open(args.output_path, "w") as f:
        for s in selected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(selected)} samples to {args.output_path}")


if __name__ == "__main__":
    main()
