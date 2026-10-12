#!/usr/bin/env python3
"""规则过滤: 时长30-300s + query与测试集词重叠≥20%"""
import json, re, random
from collections import Counter

INPUT = "/mnt/250T_ceph/zhenjunchen/data/gemini_refined_data.jsonl"
OUTPUT = "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/teacher_scores_quick/subset_10k.jsonl"
BENCH_DIR = "/mnt/250T_ceph/zhenjunchen/data/TimeLens-Bench"

# 加载测试集词表
test_words = set()
for name in ['charades-timelens', 'activitynet-timelens', 'qvhighlights-timelens']:
    with open(f'{BENCH_DIR}/{name}.json') as f:
        data = json.load(f)
    for v in data.values():
        if isinstance(v, dict) and 'queries' in v:
            for q in v['queries']:
                for w in re.findall(r'[a-z]{2,}', q.lower()):
                    test_words.add(w)

stopwords = {'the','and','with','are','his','her','from','into','out','that','this','was','for','has','its','can','you','but','not','all','one','his','her','who','how','did','get','had','been'}
test_words -= stopwords

print(f"测试集关键词: {len(test_words)}")

# 过滤
kept = []
for line in open(INPUT):
    if not line.strip():
        continue
    d = json.loads(line)
    dur = float(d.get("duration", 0))
    query = d.get("query", "").lower()

    # 规则1: 时长30-300s
    if not (30 <= dur <= 300):
        continue

    # 规则2: query词重叠≥20%
    qwords = [w for w in re.findall(r'[a-z]{2,}', query) if w not in stopwords]
    if not qwords:
        continue
    overlap = len([w for w in qwords if w in test_words]) / len(qwords)
    if overlap < 0.2:
        continue

    kept.append(line.strip())

random.seed(42)
subset = random.sample(kept, min(10000, len(kept)))

with open(OUTPUT, 'w') as f:
    for l in subset:
        f.write(l + '\n')

print(f"规则过滤后: {len(kept)}/{96270} ({len(kept)/96270*100:.1f}%)")
print(f"随机抽 10K: {len(subset)} 条")
print(f"输出: {OUTPUT}")
