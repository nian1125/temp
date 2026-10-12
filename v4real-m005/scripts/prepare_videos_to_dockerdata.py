#!/usr/bin/env python3
"""把 train_reviewed_all_merged.jsonl 所有视频复制到 dockerdata。"""
import json, os, shutil, subprocess

ANNO = "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/train_reviewed_all_merged.jsonl"
VIDEO_SHARDS = "/mnt/250T_ceph/zhenjunchen/data/TimeLens-100K/video_shards"
DST_DIR = "/dockerdata/zhenjunchen/train_videos_reviewed"
OUT_ANNO = "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/train_reviewed_all_merged_abs.jsonl"

os.makedirs(DST_DIR, exist_ok=True)

with open(ANNO) as f:
    annos = [json.loads(line) for line in f if line.strip()]

copied = 0
missing = 0
new_annos = []

for a in annos:
    vp = a["video_path"]
    fname = os.path.basename(vp)
    dst = os.path.join(DST_DIR, fname)

    # 找源文件
    src = None
    if vp.startswith("/"):
        if os.path.exists(vp):
            src = vp
        else:
            # 搜索 100K shards 中所有子目录
            for ds in os.listdir(VIDEO_SHARDS):
                candidate = os.path.join(VIDEO_SHARDS, ds, ds, fname)
                if os.path.isfile(candidate):
                    src = candidate
                    break
    else:
        parts = vp.split("/", 1)
        if len(parts) == 2:
            candidate = os.path.join(VIDEO_SHARDS, parts[0], parts[0], parts[1])
            if os.path.exists(candidate):
                src = candidate

    if src:
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied += 1
        a["video_path"] = dst
    else:
        missing += 1
        print(f"  MISS: {vp} -> (not found in shards)")
    new_annos.append(a)

with open(OUT_ANNO, "w") as f:
    for a in new_annos:
        f.write(json.dumps(a, ensure_ascii=False) + "\n")

print(f"Done. Copied: {copied}, Missing: {missing}")
print(f"New anno: {OUT_ANNO}")
