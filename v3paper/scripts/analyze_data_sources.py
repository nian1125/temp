#!/usr/bin/env python3
"""分析训练数据的来源分布。

在训练机上运行:
    python scripts/analyze_data_sources.py

默认读取 gemini_refined_data.jsonl，统计各来源占比。
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

def infer_source_from_path(video_path):
    """从 video_path 推断数据来源（如果没有 source 字段）"""
    vp = video_path.lower()
    if "charades" in vp:
        return "Charades"
    elif "activitynet" in vp or "activity_net" in vp:
        return "ActivityNet"
    elif "qvhighlights" in vp or "qvh" in vp:
        return "QVHighlights"
    elif "didemo" in vp:
        return "DiDeMo"
    elif "tacos" in vp:
        return "TACoS"
    elif "youcook" in vp:
        return "YouCook"
    else:
        return f"unknown({os.path.dirname(vp).split('/')[-1]})"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", default=None,
                        help="Path to training JSONL. Auto-detected if not given.")
    args = parser.parse_args()

    if args.anno_path is None:
        candidates = [
            os.path.join(os.environ.get("TIMELENS_ROOT", ""),
                         "data/filter-data/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl"),
            "gemini_refined_data.jsonl",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.anno_path = c
                break
        if args.anno_path is None:
            print("ERROR: Could not find annotation file. Use --anno_path.")
            return

    print(f"Reading: {args.anno_path}")

    source_counts = Counter()
    path_source_counts = Counter()
    duration_by_source = defaultdict(list)
    total = 0
    has_source_field = False
    sample_paths = defaultdict(list)

    with open(args.anno_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            total += 1

            src = d.get("source", None)
            if src:
                has_source_field = True
                source_counts[src] += 1

            vp = d.get("video_path", "")
            inferred = infer_source_from_path(vp)
            path_source_counts[inferred] += 1

            dur = d.get("duration", 0)
            if src:
                duration_by_source[src].append(dur)
            else:
                duration_by_source[inferred].append(dur)

            if len(sample_paths[inferred]) < 2:
                sample_paths[inferred].append(vp)

    print(f"\nTotal samples: {total}")

    if has_source_field:
        print(f"\n{'='*60}")
        print(f"By 'source' field:")
        print(f"{'='*60}")
        for src, cnt in source_counts.most_common():
            pct = cnt / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {src:30s} {cnt:6d} ({pct:5.1f}%) {bar}")

    print(f"\n{'='*60}")
    print(f"By video_path inference:")
    print(f"{'='*60}")
    for src, cnt in path_source_counts.most_common():
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {src:30s} {cnt:6d} ({pct:5.1f}%) {bar}")
        for sp in sample_paths[src][:1]:
            print(f"    e.g. {sp}")

    print(f"\n{'='*60}")
    print(f"Duration stats by source:")
    print(f"{'='*60}")
    key = source_counts if has_source_field else path_source_counts
    for src, _ in sorted(key.items(), key=lambda x: -x[1]):
        durs = duration_by_source[src]
        if durs:
            avg_dur = sum(durs) / len(durs)
            min_dur = min(durs)
            max_dur = max(durs)
            print(f"  {src:30s} avg={avg_dur:6.1f}s  range=[{min_dur:.0f}, {max_dur:.0f}]s")

    # Charades-specific analysis
    charades_key = None
    for k in (source_counts if has_source_field else path_source_counts):
        if "charades" in k.lower():
            charades_key = k
            break

    if charades_key:
        c_cnt = (source_counts if has_source_field else path_source_counts)[charades_key]
        print(f"\n{'='*60}")
        print(f"Charades Analysis (test domain):")
        print(f"{'='*60}")
        print(f"  Charades samples: {c_cnt}/{total} = {c_cnt/total*100:.1f}%")
        print(f"  TimeLens-Bench test: 1313 videos, 3363 queries (100% Charades)")
        print(f"  Train/Test domain match: {'Good' if c_cnt/total > 0.3 else 'LOW - consider upsampling'}")
    else:
        print(f"\n  WARNING: No Charades samples found! Check source/path patterns.")

if __name__ == "__main__":
    main()
