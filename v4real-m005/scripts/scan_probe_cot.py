#!/usr/bin/env python3
"""汇总 probe_cot 结果: 裸答案vs思考的mIoU、是否真产出推理、并dump几条CoT原文供人工判断。
用法: python3 scan_probe_cot.py <probe_cot目录>
"""
import sys, json, glob, re

d = sys.argv[1] if len(sys.argv) > 1 else "output/probe_cot"
recs = []
for f in glob.glob(f"{d}/p_*.jsonl"):
    for l in open(f):
        if l.strip():
            try: recs.append(json.loads(l))
            except Exception: pass

if not recs:
    print("还没结果。"); sys.exit(0)

def avg(xs): return sum(xs)/len(xs) if xs else 0.0

# 是否真产出推理: think输出去掉答案句后仍有实质内容(>40字)
def has_reasoning(r):
    t = r.get("think_out", "")
    body = re.sub(r"the event happens in.*", "", t, flags=re.I|re.S).strip()
    return len(body) > 40

pl = [r["plain_tiou"] for r in recs if "plain_tiou" in r]
th = [r["think_tiou"] for r in recs if "think_tiou" in r]
both = [r for r in recs if "plain_tiou" in r and "think_tiou" in r]
reasoning = [r for r in recs if has_reasoning(r)]

print(f"样本数 n={len(recs)}")
print(f"裸答案 mIoU = {avg(pl)*100:.2f}")
print(f"思考   mIoU = {avg(th)*100:.2f}   (Δ思考 = {(avg(th)-avg(pl))*100:+.2f})")
print(f"真产出推理的比例 = {len(reasoning)}/{len(recs)} = {len(reasoning)/len(recs)*100:.0f}%")
print(f"思考输出平均长度 = {avg([r.get('think_len',0) for r in recs]):.0f} 字符")

# 难子集(裸答案<0.5)
hard = [r for r in both if r["plain_tiou"] < 0.5]
if hard:
    print(f"\n[难子集 裸答案tIoU<0.5, n={len(hard)}]")
    print(f"  裸答案 mIoU = {avg([r['plain_tiou'] for r in hard])*100:.2f}")
    print(f"  思考   mIoU = {avg([r['think_tiou'] for r in hard])*100:.2f}   "
          f"(Δ = {(avg([r['think_tiou'] for r in hard])-avg([r['plain_tiou'] for r in hard]))*100:+.2f})")

# dump 几条CoT原文(混合: 思考帮了的 / 思考害了的 / 随机)
print("\n========== CoT 原文样例(人工判断推理质量) ==========")
both_sorted = sorted(both, key=lambda r: r["think_tiou"]-r["plain_tiou"])
picks = []
if both_sorted:
    picks += [("思考帮最多", both_sorted[-1]), ("思考帮第二", both_sorted[-2] if len(both_sorted)>1 else both_sorted[-1])]
    picks += [("思考害最多", both_sorted[0]), ("中间", both_sorted[len(both_sorted)//2])]
seen=set()
for tag, r in picks:
    if r["_idx"] in seen: continue
    seen.add(r["_idx"])
    print(f"\n--- [{tag}] query: {r['query']}")
    print(f"    GT={r['gt']}  裸tIoU={r.get('plain_tiou')}  思考tIoU={r.get('think_tiou')}")
    print(f"    裸答案: {r.get('plain_out','')[:120]}")
    print(f"    思考全文: {r.get('think_out','')}")

print("\n判断: (1)真产出推理比例高吗? (2)思考mIoU是否≥裸答案? (3)CoT是真从画面推还是空话/编造?")
print("→ 都OK则CoT线可行; 只吐答案/思考更差/CoT空洞 则CoT线需先解决'让模型会推理'")
