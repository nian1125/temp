#!/usr/bin/env python3
"""比较干净推理 vs 红蓝框推理，找出红蓝框收益样本。
用法: python scripts/find_redblue_gains.py --clean <path> --redblue <path> --output <path>
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--redblue", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gap_threshold", type=float, default=0.05,
                    help="tIoU 增益阈值，超过此值视为红蓝框有帮助")
    args = ap.parse_args()

    clean = {}
    for line in open(args.clean):
        d = json.loads(line)
        clean[d["_idx"]] = d

    rb = {}
    for line in open(args.redblue):
        d = json.loads(line)
        rb[d["_idx"]] = d

    stats = {"total": 0, "rb_helps": 0, "rb_hurts": 0, "same": 0}
    gains = []

    for idx in clean:
        c = clean[idx]["tiou"]
        r = rb.get(idx, {}).get("tiou", 0)
        gap = r - c
        stats["total"] += 1
        if gap > args.gap_threshold:
            stats["rb_helps"] += 1
            gains.append({
                "_idx": idx,
                "video_path": clean[idx]["video_path"],
                "query": clean[idx]["query"],
                "gt_span": clean[idx]["gt_span"],
                "tiou_clean": round(c, 4),
                "tiou_redblue": round(r, 4),
                "gap": round(gap, 4),
                "pred_clean": clean[idx].get("answer", "")[:100],
                "pred_redblue": rb.get(idx, {}).get("answer", "")[:100],
            })
        elif gap < -args.gap_threshold:
            stats["rb_hurts"] += 1
        else:
            stats["same"] += 1

    gains.sort(key=lambda x: x["gap"], reverse=True)

    with open(args.output, "w") as f:
        for g in gains:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print(f"Total: {stats['total']}")
    print(f"RedBlue helps (gap>{args.gap_threshold}): {stats['rb_helps']} ({100*stats['rb_helps']/stats['total']:.1f}%)")
    print(f"RedBlue hurts (gap<-{args.gap_threshold}): {stats['rb_hurts']} ({100*stats['rb_hurts']/stats['total']:.1f}%)")
    print(f"About same: {stats['same']}")
    print(f"\nTop 10 gains:")
    for g in gains[:10]:
        print(f"  +{g['gap']:.4f} | clean={g['tiou_clean']:.4f} rb={g['tiou_redblue']:.4f} | {g['query'][:60]}")
    print(f"\nSaved {len(gains)} help samples to {args.output}")


if __name__ == "__main__":
    main()
