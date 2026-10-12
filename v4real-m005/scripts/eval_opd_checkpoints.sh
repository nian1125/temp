#!/bin/bash
# Batch evaluation of OPD-GRPO checkpoints on TimeLens-Bench
# Usage: bash scripts/eval_opd_checkpoints.sh
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"
BASE_MODEL="/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct"

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

# Video processing params (match training config)
export min_tokens=64
export total_tokens=14336
export FPS=2

# Only evaluate Charades (fastest, most relevant)
export datasets="charades-timelens"

# Checkpoints to evaluate
declare -A CHECKPOINTS=(
    ["base"]="$BASE_MODEL"
    ["v1-kl1-step100"]="$OPD_OUTPUT/opd-numberit+red_border-kl1.0-lr1e-6-20260617-1118/checkpoint-100"
    ["v2-kl5-step100"]="$OPD_OUTPUT/opd-numberit+red_border-kl5.0-lr1e-6-20260618-0528/checkpoint-100"
    ["v3-kl2-step50"]="$OPD_OUTPUT/opd-numberit+red_border-kl2.0-lr1e-6-20260618-1550/checkpoint-50"
    ["v3-kl2-step100"]="$OPD_OUTPUT/opd-numberit+red_border-kl2.0-lr1e-6-20260618-1550/checkpoint-100"
    ["v3-kl2-step150"]="$OPD_OUTPUT/opd-numberit+red_border-kl2.0-lr1e-6-20260618-1550/checkpoint-150"
    ["v3-kl2-step200"]="$OPD_OUTPUT/opd-numberit+red_border-kl2.0-lr1e-6-20260618-1550/checkpoint-200"
)

echo "=========================================="
echo "OPD-GRPO Checkpoint Evaluation"
echo "=========================================="

RESULTS_FILE="$OPD_OUTPUT/eval_results_summary.txt"
echo "# OPD-GRPO Evaluation Results ($(date))" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for tag in "${!CHECKPOINTS[@]}"; do
    ckpt="${CHECKPOINTS[$tag]}"

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

    # Extract metrics from log
    metrics=$(grep -A5 "Completed evaluation" /tmp/eval_${tag}.log | tail -4)
    echo "" >> "$RESULTS_FILE"
    echo "## $tag" >> "$RESULTS_FILE"
    echo "Path: $ckpt" >> "$RESULTS_FILE"
    echo "$metrics" >> "$RESULTS_FILE"

    echo ""
    echo "[$tag] Done."
done

echo ""
echo "=========================================="
echo "All evaluations complete."
echo "Results saved to: $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"
