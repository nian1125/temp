#!/usr/bin/env python3
"""分析 teacher vs student 的 per-sample 边界误差 (start/end/center/duration).

输入: eval_teacher_strategies.py 产出的 shard_*.jsonl
输出: 每个样本的误差分解 + 汇总统计

Usage:
    python scripts/analyze_boundary_errors.py \
        --plain_dir output/teacher_eval/plain \
        --privileged_dir output/teacher_eval/coarse_box \
        --output_dir output/teacher_eval/analysis
"""

import argparse, json, os, glob
import numpy as np


def load_shards(shard_dir):
    """加载所有 shard 文件, 按 video_id 索引"""
    records = {}
    for f in sorted(glob.glob(os.path.join(shard_dir, "shard_*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    records[r["video_id"]] = r
    return records


def compute_errors(gt_span, pred_span, duration):
    """计算单个样本的各类边界误差."""
    gt_s, gt_e = gt_span
    pred_s, pred_e = pred_span
    gt_len = gt_e - gt_s

    start_err = pred_s - gt_s          # >0 偏晚, <0 偏早
    end_err = pred_e - gt_e            # >0 偏晚, <0 偏早
    center_err = ((pred_s + pred_e)/2 - (gt_s + gt_e)/2)  # 中心偏移
    duration_err = (pred_e - pred_s) - gt_len              # >0 预测过长

    # 归一化误差 (相对于 GT 时长)
    start_err_norm = start_err / gt_len if gt_len > 0 else 0
    end_err_norm = end_err / gt_len if gt_len > 0 else 0
    center_err_norm = center_err / gt_len if gt_len > 0 else 0
    duration_err_norm = duration_err / gt_len if gt_len > 0 else 0

    # 绝对误差
    abs_start_err = abs(start_err)
    abs_end_err = abs(end_err)
    abs_center_err = abs(center_err)

    return {
        "gt_start": gt_s, "gt_end": gt_e, "gt_duration": gt_len,
        "pred_start": pred_s, "pred_end": pred_e,
        "start_err": start_err, "end_err": end_err,
        "center_err": center_err, "duration_err": duration_err,
        "start_err_norm": start_err_norm, "end_err_norm": end_err_norm,
        "center_err_norm": center_err_norm, "duration_err_norm": duration_err_norm,
        "abs_start_err": abs_start_err, "abs_end_err": abs_end_err,
        "abs_center_err": abs_center_err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain_dir", required=True)
    parser.add_argument("--privileged_dir", required=True)
    parser.add_argument("--output_dir", default="output/teacher_eval/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plain = load_shards(args.plain_dir)
    priv  = load_shards(args.privileged_dir)
    print(f"Plain: {len(plain)} samples")
    print(f"Privileged: {len(priv)} samples")

    # 对齐样本
    common_ids = sorted(set(plain.keys()) & set(priv.keys()))
    print(f"Common: {len(common_ids)} samples")
    if len(common_ids) == 0:
        print("ERROR: no common video_ids! Check shard files.")
        return

    # 逐样本计算误差
    results = []
    for vid in common_ids:
        p = plain[vid]
        pr = priv[vid]

        gt = p["span"]
        duration = gt[1] - gt[0]

        # plain 误差
        if p["pred"]:
            err_plain = compute_errors(gt, p["pred"], duration)
            err_plain["tiou_plain"] = p["tiou"]
        else:
            err_plain = None

        # privileged 误差
        if pr["pred"]:
            err_priv = compute_errors(gt, pr["pred"], duration)
            err_priv["tiou_privileged"] = pr["tiou"]
        else:
            err_priv = None

        results.append({
            "video_id": vid,
            "query": p["query"],
            "duration": duration,
            "plain": err_plain,
            "privileged": err_priv,
            "delta_tiou": (pr["tiou"] - p["tiou"]) if (p["pred"] and pr["pred"]) else 0,
            "plain_answer": p.get("answer", ""),
            "privileged_answer": pr.get("answer", ""),
        })

    # ---- 汇总统计 ----

    # 筛选有有效预测的样本
    both_valid = [r for r in results if r["plain"] is not None and r["privileged"] is not None]
    plain_only = [r for r in results if r["plain"] is not None]
    priv_only  = [r for r in results if r["privileged"] is not None]

    print(f"\nBoth valid: {len(both_valid)}")
    print(f"Plain valid: {len(plain_only)}, Privileged valid: {len(priv_only)}")

    # 聚合指标
    metrics = {
        "plain": {"abs_start_err": [], "abs_end_err": [], "abs_center_err": [],
                   "start_err": [], "end_err": [], "center_err": [], "duration_err": [],
                   "tiou": []},
        "privileged": {"abs_start_err": [], "abs_end_err": [], "abs_center_err": [],
                        "start_err": [], "end_err": [], "center_err": [], "duration_err": [],
                        "tiou": []},
    }

    for r in both_valid:
        for key in ["plain", "privileged"]:
            e = r[key]
            for k in ["abs_start_err", "abs_end_err", "abs_center_err",
                       "start_err", "end_err", "center_err", "duration_err"]:
                metrics[key][k].append(e[k])
            if key == "plain":
                metrics[key]["tiou"].append(r["plain"]["tiou_plain"])
            else:
                metrics[key]["tiou"].append(r["privileged"]["tiou_privileged"])

    # ---- 打印汇总 ----
    print("\n" + "="*80)
    print("边界误差分析汇总 (Per-Sample Mean ± Std)")
    print("="*80)
    print(f"{'Metric':<30} {'Plain (SFT)':<30} {'Privileged (coarse 1.5x)':<30} {'Improvement':<15}")
    print("-"*105)

    for metric_name, field in [
        ("|start_err| (abs, sec)", "abs_start_err"),
        ("|end_err| (abs, sec)", "abs_end_err"),
        ("|center_err| (abs, sec)", "abs_center_err"),
        ("start_err (signed, sec)", "start_err"),
        ("end_err (signed, sec)", "end_err"),
        ("center_err (signed, sec)", "center_err"),
        ("duration_err (signed, sec)", "duration_err"),
        ("tIoU", "tiou"),
    ]:
        p_vals = metrics["plain"][field]
        pr_vals = metrics["privileged"][field]
        p_mean, p_std = np.mean(p_vals), np.std(p_vals)
        pr_mean, pr_std = np.mean(pr_vals), np.std(pr_vals)

        if "abs" in field or "tIoU" in field:
            # 绝对误差越小越好
            improvement = (p_mean - pr_mean) / max(abs(p_mean), 1e-8) * 100
            arrow = "↓" if improvement > 0 else "↑"
        else:
            # signed 看偏移方向
            improvement = (abs(p_mean) - abs(pr_mean)) / max(abs(p_mean), 1e-8) * 100 if abs(p_mean) > 0 else 0
            arrow = "↓" if improvement > 0 else "↑"

        print(f"{metric_name:<30} {p_mean:6.2f} ± {p_std:5.2f}      "
              f"{pr_mean:6.2f} ± {pr_std:5.2f}      {improvement:+5.1f}% {arrow}")

    # ---- 分桶分析：按 GT duration 分组 ----
    durations = [r["duration"] for r in both_valid]
    p25, p50, p75 = np.percentile(durations, [25, 50, 75])
    print(f"\nGT duration percentiles: P25={p25:.1f}s, P50={p50:.1f}s, P75={p75:.1f}s")

    for label, lo, hi in [("Short (<P25)", 0, p25), ("Medium (P25-P75)", p25, p75), ("Long (>P75)", p75, 1e9)]:
        subset = [r for r in both_valid if lo <= r["duration"] < hi]
        if len(subset) < 5:
            continue
        p_tiou = np.mean([r["plain"]["tiou_plain"] for r in subset])
        pr_tiou = np.mean([r["privileged"]["tiou_privileged"] for r in subset])
        p_as = np.mean([r["plain"]["abs_start_err"] for r in subset])
        pr_as = np.mean([r["privileged"]["abs_start_err"] for r in subset])
        p_ae = np.mean([r["plain"]["abs_end_err"] for r in subset])
        pr_ae = np.mean([r["privileged"]["abs_end_err"] for r in subset])
        print(f"\n  {label} (n={len(subset)}):")
        print(f"    tIoU:     plain={p_tiou:.3f} → priv={pr_tiou:.3f}  Δ={pr_tiou-p_tiou:+.3f}")
        print(f"    |start_err|: plain={p_as:.2f}s → priv={pr_as:.2f}s  Δ={pr_as-p_as:+.2f}s")
        print(f"    |end_err|:   plain={p_ae:.2f}s → priv={pr_ae:.2f}s  Δ={pr_ae-p_ae:+.2f}s")

    # ---- 保存详细数据 ----
    # 全量 per-sample
    with open(os.path.join(args.output_dir, "per_sample_errors.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总
    summary = {
        "n_samples": len(both_valid),
        "plain_tiou": float(np.mean(metrics["plain"]["tiou"])),
        "privileged_tiou": float(np.mean(metrics["privileged"]["tiou"])),
        "delta_tiou": float(np.mean([r["delta_tiou"] for r in both_valid])),
        "plain_abs_start_err": float(np.mean(metrics["plain"]["abs_start_err"])),
        "privileged_abs_start_err": float(np.mean(metrics["privileged"]["abs_start_err"])),
        "plain_abs_end_err": float(np.mean(metrics["plain"]["abs_end_err"])),
        "privileged_abs_end_err": float(np.mean(metrics["privileged"]["abs_end_err"])),
        "plain_abs_center_err": float(np.mean(metrics["plain"]["abs_center_err"])),
        "privileged_abs_center_err": float(np.mean(metrics["privileged"]["abs_center_err"])),
        "plain_start_err_signed": float(np.mean(metrics["plain"]["start_err"])),
        "privileged_start_err_signed": float(np.mean(metrics["privileged"]["start_err"])),
        "plain_end_err_signed": float(np.mean(metrics["plain"]["end_err"])),
        "privileged_end_err_signed": float(np.mean(metrics["privileged"]["end_err"])),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n详细结果已保存到: {args.output_dir}/")
    print(f"  per_sample_errors.jsonl — 每个样本的完整误差")
    print(f"  summary.json — 汇总统计")


if __name__ == "__main__":
    main()
