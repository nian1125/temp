#!/bin/bash
# v5 + v6 checkpoint evaluation on TimeLens-Bench (Charades)
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"
V5_DIR="$OPD_OUTPUT/opd-numberit+red_border-kl1.0-gw1.0-bnpo-spg1-lr1e-6-20260620-1847"
V6_DIR="$OPD_OUTPUT/opd-numberit+red_border-kl1.0-gw1.0-grpo-spg4-lr1e-6-20260620-2335"

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

declare -A CHECKPOINTS=(
    ["v5-step50"]="$V5_DIR/checkpoint-50"
    ["v5-step100"]="$V5_DIR/checkpoint-100"
    ["v5-step150"]="$V5_DIR/checkpoint-150"
    ["v5-step200"]="$V5_DIR/checkpoint-200"
    ["v6-step100"]="$V6_DIR/checkpoint-100"
    ["v6-step200"]="$V6_DIR/checkpoint-200"
    ["v6-step300"]="$V6_DIR/checkpoint-300"
    ["v6-step400"]="$V6_DIR/checkpoint-400"
)

RESULTS_FILE="$OPD_OUTPUT/eval_v5v6_results.txt"
echo "# v5 + v6 Evaluation Results ($(date))" > "$RESULTS_FILE"

for tag in v5-step50 v5-step100 v5-step150 v5-step200 v6-step100 v6-step200 v6-step300 v6-step400; do
    ckpt="${CHECKPOINTS[$tag]}"
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
echo "All v5+v6 evaluations complete."
echo "=========================================="
cat "$RESULTS_FILE"
