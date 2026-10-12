#!/usr/bin/env python3
"""
完整 Teacher vs Student 边界误差分析 (V2)
===========================================
基于 per-sample 预测，计算：
  1. 基础误差指标 (abs/signed/normalized start/end/center/duration)
  2. 配对改善指标 (gain/delta)
  3. 修正方向率 (correction direction rate)
  4. 错误类型分类 & 迁移矩阵
  5. Search vs Refine 分解
  6. 按样本属性分桶 (duration/ratio/position)
  7. 统计检验 (paired t-test, Wilcoxon, bootstrap CI)
  8. Logits 分析 (entropy, JSD, GT logprob) — 如果可用

Usage:
    python scripts/analyze_boundary_errors_v2.py \
        --plain_dir output/teacher_eval/plain \
        --privileged_dir output/teacher_eval/coarse_box \
        --output_dir output/teacher_eval/analysis_v2
"""

import argparse, json, os, glob, math
import numpy as np
from scipy import stats as scipy_stats
from collections import Counter, defaultdict


# ============================================================
# 工具函数
# ============================================================

def load_shards(shard_dir):
    records = {}
    for f in sorted(glob.glob(os.path.join(shard_dir, "shard_*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    records[r["video_id"]] = r
    return records


def compute_errors(gt_span, pred_span, video_duration):
    """计算单个样本的完整误差指标."""
    gt_s, gt_e = gt_span
    pred_s, pred_e = pred_span
    gt_len = gt_e - gt_s
    eps = 1e-8

    e = {}

    # --- 基础误差 (秒) ---
    e["start_err"] = pred_s - gt_s          # signed: >0=偏晚, <0=偏早
    e["end_err"] = pred_e - gt_e
    e["center_err"] = (pred_s + pred_e)/2 - (gt_s + gt_e)/2
    e["duration_err"] = (pred_e - pred_s) - gt_len  # >0=过宽, <0=过窄

    e["abs_start_err"] = abs(e["start_err"])
    e["abs_end_err"] = abs(e["end_err"])
    e["abs_center_err"] = abs(e["center_err"])
    e["abs_duration_err"] = abs(e["duration_err"])

    # --- 归一化误差 (相对于视频总长) ---
    if video_duration > eps:
        e["norm_start_err_by_video"] = e["abs_start_err"] / video_duration
        e["norm_end_err_by_video"] = e["abs_end_err"] / video_duration
        e["norm_center_err_by_video"] = e["abs_center_err"] / video_duration
    else:
        e["norm_start_err_by_video"] = 0
        e["norm_end_err_by_video"] = 0
        e["norm_center_err_by_video"] = 0

    # --- 归一化误差 (相对于GT事件长度) ---
    if gt_len > eps:
        e["norm_start_err_by_gt"] = e["abs_start_err"] / gt_len
        e["norm_end_err_by_gt"] = e["abs_end_err"] / gt_len
        e["norm_center_err_by_gt"] = e["abs_center_err"] / gt_len
        e["norm_duration_err_by_gt"] = e["abs_duration_err"] / gt_len
    else:
        e["norm_start_err_by_gt"] = 0
        e["norm_end_err_by_gt"] = 0
        e["norm_center_err_by_gt"] = 0
        e["norm_duration_err_by_gt"] = 0

    return e


def classify_error_type(pred_start, pred_end, gt_start, gt_end):
    """将预测区间分类为错误类型."""
    eps = 0.01  # 1% 容差
    iou_val = compute_iou([gt_start, gt_end], [pred_start, pred_end])

    if pred_end <= gt_start + eps:
        return "disjoint-left"
    if pred_start >= gt_end - eps:
        return "disjoint-right"

    if iou_val >= 0.5:
        return "good-overlap"

    if pred_start >= gt_start and pred_end <= gt_end:
        return "inside-gt"
    if pred_start <= gt_start and pred_end >= gt_end:
        return "contains-gt"
    if pred_start < gt_start and pred_end > gt_start:
        return "left-overlap"
    if pred_start < gt_end and pred_end > gt_end:
        return "right-overlap"

    return "other"


def compute_iou(a, b):
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    return max(min1 - max0, 0) / (max1 - min0) if (max1 - min0) > 0 else 0.0


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for mean."""
    data = np.array(data)
    n = len(data)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    return np.percentile(means, [alpha * 100, (1 - alpha) * 100])


def paired_effect_size(diff):
    """Cohen's d_z for paired data."""
    return np.mean(diff) / (np.std(diff) + 1e-8)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain_dir", required=True)
    parser.add_argument("--privileged_dir", required=True)
    parser.add_argument("--output_dir", default="output/teacher_eval/analysis_v2")
    parser.add_argument("--has_logits", action="store_true",
                        help="是否每条记录包含 logits/entropy 字段")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plain = load_shards(args.plain_dir)
    priv = load_shards(args.privileged_dir)
    print(f"Plain: {len(plain)}, Privileged: {len(priv)}")

    common_ids = sorted(set(plain.keys()) & set(priv.keys()))
    print(f"Common: {len(common_ids)}")
    if len(common_ids) == 0:
        print("ERROR: no common samples")
        return

    # ============================================================
    # Per-sample 计算
    # ============================================================
    per_sample = []
    for vid in common_ids:
        p = plain[vid]
        pr = priv[vid]
        gt = p["span"]
        gt_s, gt_e = gt
        gt_len = gt_e - gt_s
        video_dur = p.get("duration", gt_e)  # duration 字段

        row = {
            "video_id": vid,
            "query": p["query"],
            "gt_start": gt_s, "gt_end": gt_e,
            "gt_duration": gt_len,
            "video_duration": video_dur,
            "target_video_ratio": gt_len / video_dur if video_dur > 0 else 0,
            "gt_relative_center": ((gt_s + gt_e) / 2) / video_dur if video_dur > 0 else 0,
            "alpha": 1.5,
        }

        # Plain
        if p.get("pred"):
            row["plain_start"] = p["pred"][0]
            row["plain_end"] = p["pred"][1]
            row["plain_iou"] = compute_iou(gt, p["pred"])
            err_p = compute_errors(gt, p["pred"], video_dur)
            for k, v in err_p.items():
                row[f"plain_{k}"] = v
            row["plain_error_type"] = classify_error_type(row["plain_start"], row["plain_end"], gt_s, gt_e)
            row["plain_valid"] = True
        else:
            row["plain_valid"] = False
            row["plain_iou"] = 0.0
            row["plain_error_type"] = "invalid"

        # Privileged
        if pr.get("pred"):
            row["priv_start"] = pr["pred"][0]
            row["priv_end"] = pr["pred"][1]
            row["priv_iou"] = compute_iou(gt, pr["pred"])
            err_pr = compute_errors(gt, pr["pred"], video_dur)
            for k, v in err_pr.items():
                row[f"priv_{k}"] = v
            row["priv_error_type"] = classify_error_type(row["priv_start"], row["priv_end"], gt_s, gt_e)
            row["priv_valid"] = True
        else:
            row["priv_valid"] = False
            row["priv_iou"] = 0.0
            row["priv_error_type"] = "invalid"

        # 配对改善指标
        if row.get("plain_valid") and row.get("priv_valid"):
            row["delta_iou"] = row["priv_iou"] - row["plain_iou"]
            row["gain_start"] = row["plain_abs_start_err"] - row["priv_abs_start_err"]
            row["gain_end"] = row["plain_abs_end_err"] - row["priv_abs_end_err"]
            row["gain_center"] = row["plain_abs_center_err"] - row["priv_abs_center_err"]
            row["gain_duration"] = row["plain_abs_duration_err"] - row["priv_abs_duration_err"]

            # 修正方向
            gt_dir_start = gt_s - row["plain_start"]
            actual_move_start = row["priv_start"] - row["plain_start"]
            row["start_correct_direction"] = int(gt_dir_start * actual_move_start > 0)

            gt_dir_end = gt_e - row["plain_end"]
            actual_move_end = row["priv_end"] - row["plain_end"]
            row["end_correct_direction"] = int(gt_dir_end * actual_move_end > 0)

            # Search vs Refine 分类
            center_ok = abs(row["plain_center_err"]) / video_dur < 0.1 if video_dur > 0 else False
            search_improve = row["plain_abs_center_err"] > row["priv_abs_center_err"]
            refine_improve = (row["plain_abs_start_err"] + row["plain_abs_end_err"]) > \
                             (row["priv_abs_start_err"] + row["priv_abs_end_err"])
            if search_improve and not refine_improve:
                row["improve_type"] = "search-only"
            elif refine_improve and not search_improve and center_ok:
                row["improve_type"] = "refine-only"
            elif search_improve and refine_improve:
                row["improve_type"] = "search+refine"
            else:
                row["improve_type"] = "no-benefit"

            # 错误类型迁移
            row["error_transition"] = f"{row['plain_error_type']}→{row['priv_error_type']}"

        per_sample.append(row)

    # ============================================================
    # 筛选有效配对样本
    # ============================================================
    valid = [r for r in per_sample if r.get("plain_valid") and r.get("priv_valid")]
    print(f"\nValid paired samples: {len(valid)}")

    # ============================================================
    # 1. 汇总统计表
    # ============================================================
    report = []
    def p(title): report.append(title); print(title)

    p("\n" + "="*100)
    p("Video-OPSD: Teacher vs Student 边界误差分析")
    p("="*100)
    p(f"样本数: {len(valid)}")

    p(f"\n{'指标':<40} {'Plain (SFT)':<28} {'Privileged (coarse 1.5x)':<28} {'Δ':<12} {'改善%':<10} {'d_z':<8}")
    p("-"*130)

    for label, pkey, prkey, lower_is_better in [
        ("IoU", "plain_iou", "priv_iou", False),
        ("|start_err| (s)", "plain_abs_start_err", "priv_abs_start_err", True),
        ("|end_err| (s)", "plain_abs_end_err", "priv_abs_end_err", True),
        ("|center_err| (s)", "plain_abs_center_err", "priv_abs_center_err", True),
        ("|duration_err| (s)", "plain_abs_duration_err", "priv_abs_duration_err", True),
        ("start_err signed (s)", "plain_start_err", "priv_start_err", True),
        ("end_err signed (s)", "plain_end_err", "priv_end_err", True),
        ("center_err signed (s)", "plain_center_err", "priv_center_err", True),
        ("duration_err signed (s)", "plain_duration_err", "priv_duration_err", True),
        ("norm_start_err / video", "plain_norm_start_err_by_video", "priv_norm_start_err_by_video", True),
        ("norm_end_err / video", "plain_norm_end_err_by_video", "priv_norm_end_err_by_video", True),
        ("norm_center_err / video", "plain_norm_center_err_by_video", "priv_norm_center_err_by_video", True),
        ("norm_start_err / GT len", "plain_norm_start_err_by_gt", "priv_norm_start_err_by_gt", True),
        ("norm_end_err / GT len", "plain_norm_end_err_by_gt", "priv_norm_end_err_by_gt", True),
    ]:
        pv = np.array([r[pkey] for r in valid])
        prv = np.array([r[prkey] for r in valid])
        diff = prv - pv if not lower_is_better else pv - prv

        p_mean, p_std = np.mean(pv), np.std(pv)
        pr_mean, pr_std = np.mean(prv), np.std(prv)
        d_mean = np.mean(diff)
        imp_pct = (np.mean(diff) / max(np.mean(pv), 1e-8)) * 100
        dz = paired_effect_size(diff)

        p(f"{label:<40} {p_mean:6.2f}±{p_std:5.2f}       "
          f"{pr_mean:6.2f}±{pr_std:5.2f}       {d_mean:+6.2f}   {imp_pct:+6.1f}%  {dz:+5.2f}")

    # ============================================================
    # 2. 配对改善分布
    # ============================================================
    p("\n" + "-"*100)
    p("配对改善分布 (positive = privileged better)")
    p("-"*100)

    for name, key in [("Δ IoU", "delta_iou"), ("Gain start", "gain_start"),
                       ("Gain end", "gain_end"), ("Gain center", "gain_center"),
                       ("Gain duration", "gain_duration")]:
        vals = np.array([r[key] for r in valid])
        p_pos = np.mean(vals > 0) * 100
        p_zero = np.mean(vals == 0) * 100
        p_neg = np.mean(vals < 0) * 100
        med = np.median(vals)
        mean_v = np.mean(vals)
        ci_lo, ci_hi = bootstrap_ci(vals)
        # Wilcoxon
        try:
            w_stat, w_p = scipy_stats.wilcoxon(vals) if len(vals) > 5 else (0, 1)
        except:
            w_stat, w_p = 0, 1

        p(f"  {name:<18}: mean={mean_v:+.4f}  median={med:+.4f}  "
          f"CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
          f"Pos={p_pos:.1f}%  Zero={p_zero:.1f}%  Neg={p_neg:.1f}%  "
          f"Wilcoxon p={w_p:.4f}")

    # ============================================================
    # 3. 修正方向率
    # ============================================================
    p("\n" + "-"*100)
    p("修正方向分析")
    p("-"*100)
    sc_rate = np.mean([r["start_correct_direction"] for r in valid]) * 100
    ec_rate = np.mean([r["end_correct_direction"] for r in valid]) * 100
    p(f"  Start 朝正确方向修正: {sc_rate:.1f}%")
    p(f"  End 朝正确方向修正:   {ec_rate:.1f}%")

    # ============================================================
    # 4. Search vs Refine 分类
    # ============================================================
    p("\n" + "-"*100)
    p("Search vs Refine 分类")
    p("-"*100)
    imp_counts = Counter(r["improve_type"] for r in valid)
    for t in ["search-only", "refine-only", "search+refine", "no-benefit"]:
        cnt = imp_counts.get(t, 0)
        p(f"  {t:<20}: {cnt:5d} ({cnt/len(valid)*100:5.1f}%)")

    # ============================================================
    # 5. 错误类型迁移矩阵
    # ============================================================
    p("\n" + "-"*100)
    p("错误类型迁移矩阵 (Top 15)")
    p("-"*100)
    trans = Counter(r["error_transition"] for r in valid)
    for t, cnt in trans.most_common(15):
        p(f"  {t:<35}: {cnt:5d}")

    # ============================================================
    # 6. 按目标占比分桶
    # ============================================================
    p("\n" + "-"*100)
    p("按 target/video ratio 分桶")
    p("-"*100)
    ratios = np.array([r["target_video_ratio"] for r in valid])
    p25r, p50r, p75r = np.percentile(ratios, [25, 50, 75])

    for label, lo, hi in [("Short ratio (<P25)", 0, p25r),
                            ("Med ratio (P25-P75)", p25r, p75r),
                            ("Long ratio (>P75)", p75r, 1e9)]:
        subset = [r for r in valid if lo <= r["target_video_ratio"] < hi]
        if len(subset) < 5:
            continue
        p_iou = np.mean([r["plain_iou"] for r in subset])
        pr_iou = np.mean([r["priv_iou"] for r in subset])
        p_as = np.mean([r["plain_abs_start_err"] for r in subset])
        pr_as = np.mean([r["priv_abs_start_err"] for r in subset])
        p_ae = np.mean([r["plain_abs_end_err"] for r in subset])
        pr_ae = np.mean([r["priv_abs_end_err"] for r in subset])
        p_ac = np.mean([r["plain_abs_center_err"] for r in subset])
        pr_ac = np.mean([r["priv_abs_center_err"] for r in subset])
        p(f"\n  {label} (n={len(subset)}, ratio∈[{lo:.3f},{hi:.3f}]):")
        p(f"    IoU:        plain={p_iou:.3f} → priv={pr_iou:.3f}  Δ={pr_iou-p_iou:+.3f}")
        p(f"    |start_err|: plain={p_as:.2f}s → priv={pr_as:.2f}s  Δ={pr_as-p_as:+.2f}s")
        p(f"    |end_err|:   plain={p_ae:.2f}s → priv={pr_ae:.2f}s  Δ={pr_ae-p_ae:+.2f}s")
        p(f"    |center_err|: plain={p_ac:.2f}s → priv={pr_ac:.2f}s  Δ={pr_ac-p_ac:+.2f}s")

    # ============================================================
    # 7. 按GT相对位置分桶
    # ============================================================
    p("\n" + "-"*100)
    p("按 GT 在视频中相对位置分桶")
    p("-"*100)
    for label, lo, hi in [("前段 (0-0.33)", 0, 0.33), ("中段 (0.33-0.67)", 0.33, 0.67), ("后段 (0.67-1.0)", 0.67, 1.0)]:
        subset = [r for r in valid if lo <= r["gt_relative_center"] < hi]
        if len(subset) < 5:
            continue
        p_iou = np.mean([r["plain_iou"] for r in subset])
        pr_iou = np.mean([r["priv_iou"] for r in subset])
        p_as = np.mean([r["plain_abs_start_err"] for r in subset])
        pr_as = np.mean([r["priv_abs_start_err"] for r in subset])
        p(f"\n  {label} (n={len(subset)}):")
        p(f"    IoU:        plain={p_iou:.3f} → priv={pr_iou:.3f}  Δ={pr_iou-p_iou:+.3f}")
        p(f"    |start_err|: plain={p_as:.2f}s → priv={pr_as:.2f}s  Δ={pr_as-p_as:+.2f}s")

    # ============================================================
    # 8. 按GT时长分桶
    # ============================================================
    p("\n" + "-"*100)
    p("按 GT 事件时长分桶")
    p("-"*100)
    durations = np.array([r["gt_duration"] for r in valid])
    p25d, p50d, p75d = np.percentile(durations, [25, 50, 75])
    for label, lo, hi in [("Short (<P25)", 0, p25d),
                            ("Medium (P25-P75)", p25d, p75d),
                            ("Long (>P75)", p75d, 1e9)]:
        subset = [r for r in valid if lo <= r["gt_duration"] < hi]
        if len(subset) < 5:
            continue
        p_iou = np.mean([r["plain_iou"] for r in subset])
        pr_iou = np.mean([r["priv_iou"] for r in subset])
        p(f"\n  {label} (n={len(subset)}, dur∈[{lo:.1f},{hi:.1f}]s):")
        p(f"    IoU: plain={p_iou:.3f} → priv={pr_iou:.3f}  Δ={pr_iou-p_iou:+.3f}")

    # ============================================================
    # 9. R@IoU 阈值
    # ============================================================
    p("\n" + "-"*100)
    p("Recall@1 对比")
    p("-"*100)
    p_iou_vals = np.array([r["plain_iou"] for r in valid])
    pr_iou_vals = np.array([r["priv_iou"] for r in valid])
    n = len(valid)
    for thresh in [0.3, 0.5, 0.7]:
        p_r1 = np.mean(p_iou_vals >= thresh) * 100
        pr_r1 = np.mean(pr_iou_vals >= thresh) * 100
        p(f"  R1@{thresh}: plain={p_r1:.1f}%  privileged={pr_r1:.1f}%  Δ={pr_r1-p_r1:+.1f}%")

    p(f"\n  mIoU: plain={np.mean(p_iou_vals)*100:.2f}  privileged={np.mean(pr_iou_vals)*100:.2f}  "
      f"Δ={np.mean(pr_iou_vals - p_iou_vals)*100:+.2f}")

    # ============================================================
    # 保存
    # ============================================================
    with open(os.path.join(args.output_dir, "per_sample_errors.jsonl"), "w") as f:
        for r in per_sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(args.output_dir, "report.txt"), "w") as f:
        f.write("\n".join(report))

    # 汇总 JSON
    summary = {
        "n_valid": len(valid),
        "plain_miou": float(np.mean(p_iou_vals)),
        "priv_miou": float(np.mean(pr_iou_vals)),
        "delta_miou": float(np.mean(pr_iou_vals - p_iou_vals)),
        "r1_03": {"plain": float(np.mean(p_iou_vals >= 0.3)),
                   "priv": float(np.mean(pr_iou_vals >= 0.3))},
        "r1_05": {"plain": float(np.mean(p_iou_vals >= 0.5)),
                   "priv": float(np.mean(pr_iou_vals >= 0.5))},
        "r1_07": {"plain": float(np.mean(p_iou_vals >= 0.7)),
                   "priv": float(np.mean(pr_iou_vals >= 0.7))},
        "search_vs_refine": dict(imp_counts),
        "start_correct_rate": float(sc_rate),
        "end_correct_rate": float(ec_rate),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    p(f"\n\n详细结果已保存到: {args.output_dir}/")
    p(f"  per_sample_errors.jsonl — 每个样本的完整误差 (可用于后续所有分析)")
    p(f"  report.txt — 文本报告")
    p(f"  summary.json — JSON 汇总")


if __name__ == "__main__":
    main()
