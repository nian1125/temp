#!/usr/bin/env python3
"""2D难度分桶: student_iou × teacher_iou → 4 桶, 出 grpo+cliff 训练集。

桶定义:
  easy   s_iou > 0.7                              → 丢
  grpo   0.15 ≤ s_iou ≤ 0.7                        → 留 (GRPO主力, reward方差最大)
  cliff  s_iou < 0.15 且 teacher_iou > 0.5         → 留 (OPD蒸馏核心, teacher>>student)
  dead   s_iou < 0.15 且 teacher_iou ≤ 0.5         → 丢 (teacher也救不了)

输出: grpo.jsonl + cliff.jsonl, 每条含 w_kl_scale 字段 (cliff=2.0, grpo=1.0)

用法:
  python scripts/make_2d_difficulty_set.py \
    --pool output/teacher_scores_quick/filtered_6838_with_both_iou.jsonl \
    --out_dir output/2d_sets/
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    easy, grpo, cliff, dead = [], [], [], []

    for line in open(args.pool):
        if not line.strip():
            continue
        d = json.loads(line)
        s_iou = float(d.get("student_iou", 0))
        t_iou = float(d.get("teacher_tiou", 0))

        if s_iou > 0.7:
            easy.append(d)
        elif s_iou < 0.15:
            if t_iou > 0.5:
                d["bucket"] = "cliff"
                d["w_kl_scale"] = 2.0
                cliff.append(d)
            else:
                dead.append(d)
        else:
            d["bucket"] = "grpo"
            d["w_kl_scale"] = 1.0
            grpo.append(d)

    with open(os.path.join(args.out_dir, "grpo.jsonl"), "w") as f:
        for d in grpo:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "cliff.jsonl"), "w") as f:
        for d in cliff:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    n = len(easy) + len(grpo) + len(cliff) + len(dead)
    print(f"总数: {n}")
    print(f"  easy:  {len(easy):>4}  (丢)")
    print(f"  grpo:  {len(grpo):>4}  (留, GRPO)")
    print(f"  cliff: {len(cliff):>4}  (留, OPD蒸馏 ★, w_kl_scale=2.0)")
    print(f"  dead:  {len(dead):>4}  (丢)")
    print(f"\nGRPO →  {os.path.join(args.out_dir, 'grpo.jsonl')}  ({len(grpo)} 条)")
    print(f"Cliff → {os.path.join(args.out_dir, 'cliff.jsonl')}  ({len(cliff)} 条)")


if __name__ == "__main__":
    main()
