#!/bin/bash
# v9-lr checkpoint evaluation on TimeLens-Bench (Charades)
# v9-lr: lr=2e-6, warmup=10, 训练指标显示 Q3 开始过收敛
# 早期 checkpoint (step 50-200) 可能仍有价值
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"

# 自动检测 v9-lr 目录 (lr2e-6)
V9_DIR=$(ls -dt "$OPD_OUTPUT"/opd-numberit+red_border-kl1.0-gw1.0-grpo-spg8-lr2e-6-* 2>/dev/null | head -1)

if [ -z "$V9_DIR" ]; then
    echo "ERROR: v9-lr directory not found. Looking for opd-*-lr2e-6-* in $OPD_OUTPUT"
    ls -dt "$OPD_OUTPUT"/opd-*lr2e* 2>/dev/null
    exit 1
fi

echo "Found v9-lr directory: $V9_DIR"
echo "Checkpoints:"
ls -d "$V9_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

RESULTS_FILE="$OPD_OUTPUT/eval_v9lr_results.txt"
echo "# v9-lr Evaluation Results ($(date))" > "$RESULTS_FILE"

# 评测早期 checkpoint (过收敛前) + 部分后期对照
for step in 50 100 150 200 250 300 400 500 750 999; do
    tag="v9lr-step${step}"
    ckpt="$V9_DIR/checkpoint-${step}"
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
echo "v9-lr evaluation complete."
echo "=========================================="
cat "$RESULTS_FILE"
