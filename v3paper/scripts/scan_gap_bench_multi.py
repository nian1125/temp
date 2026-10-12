#!/usr/bin/env python3
"""聚合 Charades-Bench 并行gap结果(benchg-*/g_*.jsonl), 按_idx对齐, 出全bench+难子集表。
用法(primus登录节点): python3 scan_gap_bench_multi.py /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/gap
"""
import sys, json, glob, os

base = sys.argv[1] if len(sys.argv) > 1 else "output/gap"
# 合并所有 benchg-* 的每样本记录, 按 _idx 对齐
samples = {}   # _idx -> {"tiou":{...}, "kl":{...}}
for d in sorted(glob.glob(f"{base}/benchg-*")):
    for f in glob.glob(f"{d}/g_*.jsonl"):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            idx = r.get("_idx")
            s = samples.setdefault(idx, {"tiou": {}, "kl": {}})
            for k, v in r.get("tiou", {}).items():
                if v is not None:
                    s["tiou"][k] = v
            for k, v in r.get("kl", {}).items():
                if v is not None:
                    s["kl"][k] = v

recs = list(samples.values())
strats = [s.strip() for s in
          "numberit,coarse_box,coarse_box_hint,coarse_text,numberit+coarse_box,"
          "timestamp,timestamp+coarse_box,red_border,numberit+red_border,gt_text".split(",")]

def table(subset, title):
    print(f"\n===== {title} (n={len(subset)}) =====")
    pl = [r['tiou']['plain'] for r in subset if r['tiou'].get('plain') is not None]
    pm = sum(pl)/len(pl)*100 if pl else 0
    print(f"{'strategy':<22}{'n':>5}{'mIoU':>8}{'Δ(有用)':>10}{'边界KL':>10}")
    print(f"{'plain':<22}{len(pl):>5}{pm:>8.2f}{'(基准)':>10}{'—':>10}")
    rows = []
    for s in strats:
        pairs = [(r['tiou']['plain'], r['tiou'][s]) for r in subset
                 if r['tiou'].get('plain') is not None and r['tiou'].get(s) is not None]
        if not pairs:
            continue
        d = sum(q-p for p, q in pairs)/len(pairs)*100
        pr = sum(q for _, q in pairs)/len(pairs)*100
        kls = [r['kl'][s] for r in subset if r.get('kl', {}).get(s) is not None]
        kl = sum(kls)/len(kls) if kls else None
        rows.append((s, len(pairs), pr, d, kl))
    for s, n, pr, d, kl in sorted(rows, key=lambda x: -x[3]):
        klc = f"{kl:.4f}" if kl is not None else "n/a"
        print(f"{s:<22}{n:>5}{pr:>8.2f}{d:>+10.2f}{klc:>10}")

print(f"对齐样本数 n={len(recs)}")
table(recs, "全 Charades-Bench")
for thr in (0.5, 0.3):
    hard = [r for r in recs if r['tiou'].get('plain') is not None and r['tiou']['plain'] < thr]
    table(hard, f"学生低×teacher高: 难子集 plain_tIoU<{thr}")
print("\n判据: 重点看难子集——coarse在学生做不好处Δ是否转正(易样本上的'硬墙'是否被推翻)")
