#!/usr/bin/env python3
"""修正 gemini_refined_data.jsonl 里的 video_path 为训练机实际路径"""
import json, os, shutil

INPUT = "/mnt/250T_ceph/zhenjunchen/data/gemini_refined_data.jsonl"
OUTPUT = "/mnt/250T_ceph/zhenjunchen/data/gemini_refined_data_fixed.jsonl"
ROOT = "/mnt/250T_ceph/zhenjunchen/data/TimeLens-100K/video_shards"

fixed = 0
missing = 0
with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        if not line.strip():
            continue
        d = json.loads(line)
        old = d["video_path"]  # e.g. data/TimeLens-100K/videos/hirest/xxx.mp4
        # Fix: video_shards/{source}/{source}/{video_id}.mp4
        parts = old.split("/")
        if len(parts) >= 3:
            source = parts[-2]  # hirest / cosmo_cap / didemo / ...
            video_id = parts[-1]  # xxx.mp4
            new = os.path.join(ROOT, source, source, video_id)
            if os.path.exists(new):
                d["video_path"] = new
                fixed += 1
            else:
                missing += 1
        fout.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"修正完成: {fixed} 条路径可用, {missing} 条缺失")
print(f"输出: {OUTPUT}")
if missing == 0:
    shutil.move(OUTPUT, INPUT)
    print(f"已覆盖原文件: {INPUT}")
