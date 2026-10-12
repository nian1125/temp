#!/usr/bin/env python3
"""扫描所有primus log, 提取特权predictor(measure_gap)结果, 拼成对比表 + 配对分析。
用法: python3 scan_gap_results.py [log_dir]
"""
import sys, os, re, glob

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "/cpfs01/user/chenzhaoyu.czy/code/Temporal-ZwZ/log"

def strip_prefix(line):
    # 去掉 "[2026-..] [node: x.x.x.x]" 前缀
    return re.sub(r"^\[[^\]]*\]\s*\[node:[^\]]*\]", "", line).strip()

# strategy -> (mIoU, delta, kl, n)
results = {}
for f in sorted(glob.glob(f"{LOG_DIR}/**/primus*.log", recursive=True)):
    try:
        txt = open(f, errors="ignore").read()
    except Exception:
        continue
    if "Δ(有用)" not in txt and "strategies=plain" not in txt:
        continue
    for raw in txt.splitlines():
        line = strip_prefix(raw)
        # 数据行: <strategy> <n> <mIoU> <Δ或(基准)> <KL或—>
        m = re.match(r"^([a-zA-Z][\w+]*)\s+(\d+)\s+(\d+\.\d+)\s+([+-]?\d+\.\d+|\(基准\))\s+([\d.]+|—|n/a)?\s*$", line)
        if not m:
            continue
        strat, n, miou, delta, kl = m.groups()
        if strat == "plain":
            continue
        try:
            d = float(delta)
        except (TypeError, ValueError):
            continue
        kl_v = None
        if kl and kl not in ("—", "n/a"):
            try: kl_v = float(kl)
            except ValueError: pass
        # 保留最新(覆盖), 取最大n的那条
        prev = results.get(strat)
        if prev is None or int(n) >= prev[3]:
            results[strat] = (float(miou), d, kl_v, int(n))

if not results:
    print("还没有任何gap结果(log里没找到汇总表)。")
    sys.exit(0)

# 完整表(按KL升序, KL=可迁移性, 越低越好)
print(f"{'策略':<24}{'n':>5}{'mIoU':>8}{'Δ(有用)':>10}{'边界KL(可迁移)':>16}")
print("-" * 63)
for s, (m, d, kl, n) in sorted(results.items(), key=lambda x: (x[1][2] is None, x[1][2] or 9e9)):
    klc = f"{kl:.4f}" if kl is not None else "n/a"
    print(f"{s:<24}{n:>5}{m:>8.2f}{d:>+10.2f}{klc:>16}")

# 配对分析: coarse(辅助) vs 精确(泄漏)
pairs = [
    ("文本", "coarse_text", "gt_text"),
    ("v8c", "numberit+coarse_box", "numberit+red_border"),
    ("框",  "coarse_box", "red_border"),
    ("时间戳", "timestamp+coarse_box", "timestamp"),
]
print("\n=== 配对分析: coarse(辅助推理) vs 精确(泄漏) ===")
print("命题: coarse版 Δ别崩 且 KL明显更低 = 模糊指向比精确泄漏更适合蒸馏")
for name, coarse, leaky in pairs:
    c, l = results.get(coarse), results.get(leaky)
    if not c or not l:
        miss = [x for x, r in [(coarse, c), (leaky, l)] if not r]
        print(f"[{name}] 待: {', '.join(miss)}")
        continue
    ck, lk = c[2], l[2]
    klcmp = (f"KL {ck:.4f} vs {lk:.4f} ({'✓更低' if ck is not None and lk is not None and ck < lk else '✗未更低'})"
             if ck is not None and lk is not None else "KL缺")
    dcmp = f"Δ {c[1]:+.2f} vs {l[1]:+.2f}"
    print(f"[{name}] {coarse} vs {leaky}: {dcmp} | {klcmp}")

print(f"\n已收集 {len(results)}/10 条 (numberit,coarse_box,coarse_box_hint,coarse_text,"
      "numberit+coarse_box,timestamp+coarse_box,timestamp,red_border,numberit+red_border,gt_text)")
