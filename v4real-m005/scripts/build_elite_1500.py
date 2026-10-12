#!/usr/bin/env python3
"""从已评分结果精选 1500 条（扩容版）

放宽阈值 + 均衡采样:
  - 大源 (cosmo_cap/didemo/internvid_vtime): tIoU 0.25-0.85
  - 小源 (hirest*/queryd): tIoU >= 0.15, 全收
"""
import json, random
from collections import defaultdict, Counter

SCORES_DIR = "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/teacher_scores_quick"
SUBSET = "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/teacher_scores_quick/subset_10k.jsonl"
OUT = "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/teacher_scores_quick/elite_1500.jsonl"

random.seed(42)

# 1. 加载 teacher 分数
scores = {}
for s in range(4):
    try:
        for line in open(f"{SCORES_DIR}/teacher_scores_{s}.jsonl"):
            d = json.loads(line)
            scores[d["_index"]] = d["teacher_tiou"]
    except: pass
print(f"已评分: {len(scores)}")

# 2. 加载标注
annos = {}
with open(SUBSET) as f:
    for i, line in enumerate(f):
        if line.strip():
            annos[i] = json.loads(line)

# 3. 过滤分桶
big_srcs = {"cosmo_cap", "didemo", "internvid_vtime"}
buckets = defaultdict(list)

for idx, tiou in scores.items():
    if idx not in annos: continue
    src = annos[idx].get("source", "unknown")
    if src in big_srcs:
        if 0.25 <= tiou <= 0.85:
            buckets[src].append(idx)
    else:
        if tiou >= 0.15:
            buckets[src].append(idx)

print("各源候选数:")
for src in sorted(buckets.keys()):
    print(f"  {src}: {len(buckets[src])}")

# 4. 采样: 大源约 450 每条, 小源全收
selected = []
per_big = 420
for src, pool in sorted(buckets.items()):
    if src in big_srcs:
        n = min(per_big, len(pool))
        selected.extend(random.sample(pool, n))
    else:
        selected.extend(pool)

selected = random.sample(selected, min(1500, len(selected)))

# 5. 写输出
with open(OUT, "w") as f:
    for idx in selected:
        anno = annos[idx]
        anno["teacher_tiou"] = round(scores[idx], 4)
        f.write(json.dumps(anno, ensure_ascii=False) + "\n")

# 6. 统计
tious_sel = [scores[i] for i in selected]
durs = [float(annos[i].get("duration", 0)) for i in selected]
srcs = [annos[i].get("source", "?") for i in selected]
print(f"\n输出 {len(selected)} 条")
print(f"teacher_tIoU: 中位={sorted(tious_sel)[len(tious_sel)//2]:.2f}  范围=[{min(tious_sel):.2f},{max(tious_sel):.2f}]")
print(f"时长: 中位={sorted(durs)[len(durs)//2]:.0f}s  范围=[{min(durs):.0f},{max(durs):.0f}]s")
print(f"来源: {dict(sorted(Counter(srcs).items()))}")
