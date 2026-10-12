#!/usr/bin/env python3
"""从 timelens-100k.jsonl 随机采样 N 个 query-span 对，展平为标准格式。"""
import json, random, sys

INPUT = "/data/workspace/候选数据/log/timelens-100k.jsonl"
OUTPUT = "/data/workspace/候选数据/log/sampled_2500.jsonl"
N = 2500
SEED = 42

random.seed(SEED)

# 1. 展平所有 samples
all_samples = []
for line in open(INPUT):
    d = json.loads(line)
    for evt in d["events"]:
        span = evt["span"][0] if isinstance(evt["span"], list) else evt["span"]
        all_samples.append({
            "video_path": d["video_path"],
            "duration": d["duration"],
            "query": evt["query"],
            "span": span,
        })

print(f"Total samples: {len(all_samples)}")

# 2. 随机采样
sampled = random.sample(all_samples, min(N, len(all_samples)))
print(f"Sampled: {len(sampled)}")

# 3. 输出（与 charades-timelens 格式兼容）
with open(OUTPUT, "w") as f:
    for s in sampled:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Saved: {OUTPUT}")
print("Done.")
