#!/bin/bash
# numberit teacher 消融实验评测 (TimeLens-Bench)
# 训练输出: opd-numberit-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260625-1345
# 配置: teacher_strategy=numberit, 过滤数据6838条, 200 steps
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"
TRAIN_DIR="$OPD_OUTPUT/opd-numberit-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260625-1345"

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

if [ ! -d "$TRAIN_DIR" ]; then
    echo "[ERROR] numberit training output not found: $TRAIN_DIR"
    ls -d "$OPD_OUTPUT"/opd-numberit* 2>/dev/null || echo "No numberit dirs found"
    exit 1
fi

RESULTS_FILE="$OPD_OUTPUT/eval_teacher_numberit_results.txt"
echo "# numberit Teacher Ablation Eval ($(date))" > "$RESULTS_FILE"

for step in 175 177 150 125 200 100 75 50 25; do
    tag="numberit-step${step}"
    ckpt="$TRAIN_DIR/checkpoint-${step}"

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
echo "All numberit evaluations complete."
echo "=========================================="
cat "$RESULTS_FILE"
