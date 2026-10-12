#!/bin/bash
# v8a-short + v8b-short checkpoint evaluation on TimeLens-Bench (Charades)
# v8a-short: no gate, 200 steps, save every 25 steps
# v8b-short: gate=True, 200 steps, save every 25 steps
# 共 16 个 checkpoint (每个实验 8 个: step 25,50,75,100,125,150,175,200)
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"
V8A_SHORT_DIR="$OPD_OUTPUT/opd-numberit+red_border-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260624-1337"
V8B_SHORT_DIR="$OPD_OUTPUT/opd-numberit+red_border-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260624-1340"

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

RESULTS_FILE="$OPD_OUTPUT/eval_v8_short_results.txt"
echo "# v8a-short + v8b-short Evaluation Results ($(date))" > "$RESULTS_FILE"

TAGS=()
declare -A CHECKPOINTS

for step in 25 50 75 100 125 150 175 200; do
    tag="v8a-short-step${step}"
    TAGS+=("$tag")
    CHECKPOINTS["$tag"]="$V8A_SHORT_DIR/checkpoint-${step}"
done

for step in 25 50 75 100 125 150 175 200; do
    tag="v8b-short-step${step}"
    TAGS+=("$tag")
    CHECKPOINTS["$tag"]="$V8B_SHORT_DIR/checkpoint-${step}"
done

for tag in "${TAGS[@]}"; do
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
echo "All v8-short evaluations complete."
echo "=========================================="
cat "$RESULTS_FILE"
