#!/bin/bash
# Full filtered (6838 samples) checkpoint 评测 (TimeLens-Bench)
# 500 steps, save_steps=25 → 20 checkpoints
# Reward peak at step 175-200, ordered by priority
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"
EXP_DIR="$OPD_OUTPUT/opd-numberit+red_border-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260625-0158"

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

if [ ! -d "$EXP_DIR" ]; then
    echo "[ERROR] Full filtered output not found: $EXP_DIR"
    ls -d "$OPD_OUTPUT"/opd-numberit* 2>/dev/null
    exit 1
fi

RESULTS_FILE="$OPD_OUTPUT/eval_full_filtered_results.txt"
echo "# Full Filtered (6838) Evaluation Results ($(date))" > "$RESULTS_FILE"

# Priority order: peak reward region first, then early, then late
for step in 175 200 150 125 225 100 250 75 275 50 300 325 350 375 400 425 450 475 500 25; do
    tag="full-step${step}"
    ckpt="$EXP_DIR/checkpoint-${step}"

    if [ ! -d "$ckpt" ]; then
        echo "[SKIP] $tag: $ckpt not found"
        echo "[$tag] SKIPPED" >> "$RESULTS_FILE"
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
echo "All Full filtered evaluations complete."
echo "=========================================="
cat "$RESULTS_FILE"
