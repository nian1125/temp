"""拎出 step177 突破cycle(generation cycle 22)的8条样本 + student/teacher iou。
在训练机跑:
    cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
    python scripts/find_step177_samples.py
⚠️ 用 RandomSampler(seed=42) 复现, 与真实 RepeatRandomSampler+DDP 分片可能有偏差。
"""
import json, torch, statistics as st
from torch.utils.data import RandomSampler

ANNO = "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl"
TEACH = "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/teacher_scores_all.jsonl"

samples = [json.loads(l) for l in open(ANNO) if l.strip()]
teach = {}
for l in open(TEACH):
    if l.strip():
        r = json.loads(l); sp = r["span"]
        teach[(r["query"].strip(), round(float(sp[0]), 1), round(float(sp[1]), 1))] = r["teacher_tiou"]

g = torch.Generator(); g.manual_seed(42)
all_idx = list(RandomSampler(range(len(samples)), generator=g))

for cycle in [21, 22, 23]:   # step169 / 177 / 185
    step = cycle * 8 + 1
    batch = all_idx[cycle * 8: cycle * 8 + 8]
    print(f"\n=== step{step} (cycle {cycle}) ===")
    bi, ti = [], []
    for j, idx in enumerate(batch):
        s = samples[idx]; sp = s["span"][0]
        t = teach.get((s["query"].strip(), round(float(sp[0]), 1), round(float(sp[1]), 1)), None)
        bi.append(s.get("iou", 0)); ti.append(t if t is not None else 0)
        print(f"  {j+1}. base_iou={s.get('iou',0):.2f} teacher_tiou={t if t is None else round(t,2)} "
              f"dur={s['duration']:.0f}s | {s['query'][:55]}")
    print(f"  base_iou 均值={st.mean(bi):.3f} (≥0.8: {sum(1 for x in bi if x>=0.8)}/8) | "
          f"teacher_tiou 均值={st.mean(ti):.3f}")
