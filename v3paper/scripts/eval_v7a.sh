#!/bin/bash
# v7a 评估: 精选数据 + 纯 JSD (bnpo, 无 gate)
# 用法: bash scripts/eval_v7a.sh
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"

# ================================================
# !! 请填入正确的目录名 (运行下面命令查看):
# ls -lt $OPD_OUTPUT | grep bnpo-spg1
# v7a 是最早的那个 bnpo-spg1 目录 (无 gate)
# ================================================
V7A_DIR="$OPD_OUTPUT/opd-numberit+red_border-kl1.0-gw1.0-bnpo-spg1-lr1e-6-20260623-0708"

# 自动检测: 如果未填写, 尝试自动查找
if [ "$V7A_DIR" = "$OPD_OUTPUT/TO_BE_FILLED" ]; then
    echo "请先确认 v7a 目录。当前 bnpo-spg1 目录:"
    ls -dt "$OPD_OUTPUT"/opd-numberit+red_border-kl1.0-gw1.0-bnpo-spg1-lr1e-6-* 2>/dev/null
    echo ""
    echo "v7a 是没有 teacher_quality_gate 的那个 (较早的时间戳)"
    echo "请编辑此脚本, 将 V7A_DIR 设为正确路径"
    exit 1
fi

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

RESULTS_FILE="$OPD_OUTPUT/eval_v7a_results.txt"
echo "# v7a Evaluation Results ($(date))" > "$RESULTS_FILE"

for step in 10 20 30 40 50 60 70 80 90 100; do
    tag="v7a-step${step}"
    ckpt="$V7A_DIR/checkpoint-${step}"
    if [ ! -d "$ckpt" ]; then
        echo "[SKIP] $tag: $ckpt not found"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: $tag"
    echo "  Path: $ckpt"
    echo "=========================================="

    export model_path="$ckpt"
    export pred_path="$OPD_OUTPUT/eval_preds"

    bash scripts/eval_timelens_bench.sh 2>&1 | tee /tmp/eval_${tag}.log

    metrics=$(grep -A5 "Completed evaluation" /tmp/eval_${tag}.log | tail -4)
    echo "" >> "$RESULTS_FILE"
    echo "## $tag" >> "$RESULTS_FILE"
    echo "Path: $ckpt" >> "$RESULTS_FILE"
    echo "$metrics" >> "$RESULTS_FILE"
done

echo ""
echo "=========================================="
echo "v7a evaluation complete."
echo "=========================================="
cat "$RESULTS_FILE"
