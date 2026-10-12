#!/usr/bin/env python3
"""复刻TimeLens的Gaussian IoU Curriculum难度采样: 从全量池(含'iou'+'duration')抽hard子集。
按duration分9桶, 桶内按 weight=exp(-0.5*((iou-mean)/std)^2) 加权不放回抽样。
挑"难但可学"(base IoU≈mean)的样本。输出原始记录(字段不变)。

用法(primus登录节点, 纯CPU):
  python3 difficulty_sample.py \
    --pool /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/tl_filter/gemini_iou_full.jsonl \
    --out  /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/hard_2500.jsonl \
    --target_size 2500 --mean 0.05 --std 0.2 --seed 42
"""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="全量池jsonl, 每行需含 iou 和 duration")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_size", type=int, default=2500)
    ap.add_argument("--mean", type=float, default=0.05)
    ap.add_argument("--std", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inverse_density", action="store_true",
                    help="额外按IoU直方图密度的倒数加权(拉平各IoU段, =TimeLens fixed_gaussian_sampling)")
    args = ap.parse_args()

    annos = []
    for line in open(args.pool):
        if line.strip():
            annos.append(json.loads(line))
    print(f"全量池 n={len(annos)}")
    if not annos:
        return
    print(f"首条字段: {list(annos[0].keys())}")
    if "iou" not in annos[0]:
        raise ValueError("池里没有 'iou' 字段! 难度采样需要每样本的base/SFT模型IoU。")
    if "duration" not in annos[0]:
        print("[WARN] 没有 'duration', 退回不分桶的全局高斯采样。")

    rng = np.random.default_rng(args.seed)
    # duration分桶(=TimeLens): [(0,30),(30,60),...,(210,240),(240,inf)]
    ranges = [(i, i + 30) for i in range(0, 240, 30)] + [(240, float("inf"))]
    per = int(args.target_size / len(ranges))

    has_dur = "duration" in annos[0]
    if has_dur:
        buckets = {r: [] for r in ranges}
        for i, a in enumerate(annos):
            d = float(a.get("duration", 0))
            for r in ranges:
                if r[0] <= d <= r[1]:
                    buckets[r].append(i); break
    else:
        buckets = {("all",): list(range(len(annos)))}
        per = args.target_size

    def gauss_sample(indices, k):
        if len(indices) == 0 or k <= 0:
            return []
        k = min(k, len(indices))
        iou = np.array([float(annos[i]["iou"]) for i in indices], dtype=np.float64)
        w = np.exp(-0.5 * ((iou - args.mean) / args.std) ** 2)
        if args.inverse_density:
            counts, edges = np.histogram(iou, bins=20, range=(0, 1))
            bi = np.clip(np.digitize(iou, bins=edges), 1, 20) - 1
            w = w * (1.0 / (counts + 1e-6))[bi]
        w = w / w.sum()
        return rng.choice(indices, size=k, replace=False, p=w).tolist()

    kept = []
    for r, idxs in buckets.items():
        sel = gauss_sample(idxs, per)
        kept.extend(sel)
        if idxs:
            ious = [float(annos[i]["iou"]) for i in sel]
            print(f"桶{r}: 池{len(idxs)} → 抽{len(sel)}  (抽中IoU均值{np.mean(ious):.3f})" if sel else f"桶{r}: 池{len(idxs)} → 抽0")

    with open(args.out, "w") as f:
        for i in kept:
            f.write(json.dumps(annos[i], ensure_ascii=False) + "\n")
    all_iou = [float(annos[i]["iou"]) for i in kept]
    print(f"\n共抽 {len(kept)} 条 → {args.out}")
    print(f"抽中样本IoU: 均值{np.mean(all_iou):.3f} 中位{np.median(all_iou):.3f} "
          f"(对照: 全量池IoU均值{np.mean([float(a['iou']) for a in annos]):.3f}) ← 应明显更低(更难)")


if __name__ == "__main__":
    main()
