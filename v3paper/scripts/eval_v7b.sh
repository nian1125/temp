#!/bin/bash
# v7b 评估: 精选数据 + JSD + Teacher Quality Gate (bnpo, 有 gate)
# 用法: bash scripts/eval_v7b.sh
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"

# ================================================
# !! 请填入正确的目录名 (运行下面命令查看):
# ls -lt $OPD_OUTPUT | grep bnpo-spg1
# v7b 是较晚的那个 bnpo-spg1 目录 (有 gate)
# 第一次崩溃重启, 所以可能有两个时间戳的目录
# ================================================
V7B_DIR="$OPD_OUTPUT/opd-numberit+red_border-kl1.0-gw1.0-bnpo-spg1-lr1e-6-20260623-0832"

if [ "$V7B_DIR" = "$OPD_OUTPUT/TO_BE_FILLED" ]; then
    echo "请先确认 v7b 目录。当前 bnpo-spg1 目录:"
    ls -dt "$OPD_OUTPUT"/opd-numberit+red_border-kl1.0-gw1.0-bnpo-spg1-lr1e-6-* 2>/dev/null
    echo ""
    echo "v7b 是有 teacher_quality_gate 的那个 (较晚的时间戳, 可能有两个)"
    echo "检查哪个有 checkpoint-100: ls */checkpoint-100"
    echo "请编辑此脚本, 将 V7B_DIR 设为正确路径"
    exit 1
fi

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

RESULTS_FILE="$OPD_OUTPUT/eval_v7b_results.txt"
echo "# v7b Evaluation Results ($(date))" > "$RESULTS_FILE"

for step in 10 20 30 40 50 60 70 80 90 100; do
    tag="v7b-step${step}"
    ckpt="$V7B_DIR/checkpoint-${step}"
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
echo "v7b evaluation complete."
echo "=========================================="
cat "$RESULTS_FILE"
