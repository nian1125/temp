#!/bin/bash
# v8c-short checkpoint 评测 (TimeLens-Bench)
# 第三个 seed, numberit+red_border, lr=1e-6, 200 steps
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"
V8C_DIR="$OPD_OUTPUT/opd-numberit+red_border-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260624-1913"

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

if [ ! -d "$V8C_DIR" ]; then
    echo "[ERROR] v8c-short output not found: $V8C_DIR"
    ls -d "$OPD_OUTPUT"/opd-numberit* 2>/dev/null
    exit 1
fi

RESULTS_FILE="$OPD_OUTPUT/eval_v8c_short_results.txt"
echo "# v8c-short Evaluation Results ($(date))" > "$RESULTS_FILE"

for step in 25 50 75 100 125 150 175 200; do
    tag="v8c-short-step${step}"
    ckpt="$V8C_DIR/checkpoint-${step}"

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
echo "All v8c-short evaluations complete."
echo "=========================================="
cat "$RESULTS_FILE"
