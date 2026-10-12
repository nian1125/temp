#!/usr/bin/env python3
"""从已评分的 teacher 结果中精选 500 条训练样本（单维度版，无需 student_iou）

选择策略:
  1. teacher_tIoU 在 0.3-0.7 区间（可蒸馏，但非 trivial）
  2. 数据源均衡采样（5 源各约 100 条）
  3. 时长分桶平衡（3 桶）
"""
import json, random
from collections import defaultdict, Counter

SCORES_DIR = "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/teacher_scores_quick"
SUBSET = "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/teacher_scores_quick/subset_10k.jsonl"
OUT = "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/teacher_scores_quick/elite_500.jsonl"

random.seed(42)

# 1. 加载 teacher 分数 (index → tiou)
scores = {}
for s in range(4):
    try:
        for line in open(f"{SCORES_DIR}/teacher_scores_{s}.jsonl"):
            if line.strip():
                d = json.loads(line)
                scores[d["_index"]] = d["teacher_tiou"]
    except:
        pass
print(f"已评分: {len(scores)}")

# 2. 加载标注 (index → anno)    
annos = {}
with open(SUBSET) as f:
    for i, line in enumerate(f):
        if line.strip():
            annos[i] = json.loads(line)

# 3. 过滤 + 分桶 (hirest 系列放宽阈值)
buckets = defaultdict(list)
for idx, tiou in scores.items():
    if idx not in annos:
        continue
    src = annos[idx].get("source", "unknown")
    # hirest 系列放宽: 本来就少, 全部可用
    if src.startswith("hirest"):
        if tiou >= 0.2:
            buckets[src].append(idx)
    else:
        if 0.3 <= tiou <= 0.7:
            buckets[src].append(idx)

print("各源候选数:")
for src in sorted(buckets.keys()):
    print(f"  {src}: {len(buckets[src])}")

# 4. 均衡采样 (大源约 125, 小源全收)
selected = []
per_big = 400 // 3  # cosmo_cap, internvid_vtime, queryd+hirest 大源
for src, pool in sorted(buckets.items()):
    if len(pool) <= 20:
        selected.extend(pool)  # 小源全收
    else:
        n = min(per_big + 25, len(pool))
        selected.extend(random.sample(pool, n))
selected = random.sample(selected, min(500, len(selected)))

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
print(f"teacher_tIoU: {min(tious_sel):.2f}-{max(tious_sel):.2f} (中位 {sorted(tious_sel)[len(tious_sel)//2]:.2f})")
print(f"时长: {min(durs):.0f}s-{max(durs):.0f}s (中位 {sorted(durs)[len(durs)//2]:.0f}s)")
print(f"来源: {dict(sorted(Counter(srcs).items()))}")
