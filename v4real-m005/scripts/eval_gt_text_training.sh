#!/bin/bash
# gt_text teacher 训练 checkpoint 评测 (TimeLens-Bench)
# 训练输出: opd-gt_text-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260624-1619
# 配置: teacher_strategy=gt_text, batch=8, spg=8, lr=1e-6, 200 steps, save every 25
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"
GT_TEXT_DIR="$OPD_OUTPUT/opd-gt_text-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260624-1619"

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

if [ ! -d "$GT_TEXT_DIR" ]; then
    echo "[ERROR] gt_text training output not found: $GT_TEXT_DIR"
    echo "Listing available dirs:"
    ls -d "$OPD_OUTPUT"/opd-gt_text* 2>/dev/null || echo "No gt_text dirs found"
    exit 1
fi

RESULTS_FILE="$OPD_OUTPUT/eval_gt_text_training_results.txt"
echo "# gt_text Training Checkpoint Evaluation ($(date))" > "$RESULTS_FILE"

for step in 25 50 75 100 125 150 175 200; do
    tag="gt_text-step${step}"
    ckpt="$GT_TEXT_DIR/checkpoint-${step}"

    if [ ! -d "$ckpt" ]; then
        echo "[SKIP] $tag: $ckpt not found"
        echo "[$tag] SKIPPED — path not found: $ckpt" >> "$RESULTS_FILE"
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

    echo "" >> "$RESULTS_FILE"
    echo "## $tag" >> "$RESULTS_FILE"
    echo "Path: $ckpt" >> "$RESULTS_FILE"
    grep -E "IOU 0\.[357]:|mIOU:" /tmp/eval_${tag}.log | grep -v "Warning" | tail -4 >> "$RESULTS_FILE"

    echo ""
    echo "[$tag] Done."
done

echo ""
echo "=========================================="
echo "All gt_text training evaluations complete."
echo "=========================================="
cat "$RESULTS_FILE"
