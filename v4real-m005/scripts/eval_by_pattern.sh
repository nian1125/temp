#!/bin/bash
# ============================================================
# 参数化评测: 按 glob 模式自动定位训练输出目录, 评测全部 checkpoint
# 用法: bash scripts/eval_by_pattern.sh "<glob_pattern>" "<tag>"
# 例:   bash scripts/eval_by_pattern.sh "opd-numberit-kl0.0-*-klsweep-*" "kl0.0"
# 自动选最新匹配目录(按修改时间)。
# ============================================================
set -e

PATTERN="$1"
TAG="$2"
if [ -z "$PATTERN" ] || [ -z "$TAG" ]; then
    echo "用法: bash scripts/eval_by_pattern.sh \"<glob_pattern>\" \"<tag>\""
    exit 1
fi

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"

# 自动定位最新匹配目录
TRAIN_DIR="$(ls -dt "$OPD_OUTPUT"/$PATTERN 2>/dev/null | head -1)"
if [ -z "$TRAIN_DIR" ] || [ ! -d "$TRAIN_DIR" ]; then
    echo "[ERROR] 没找到匹配目录: $OPD_OUTPUT/$PATTERN"
    ls -d "$OPD_OUTPUT"/opd-* 2>/dev/null | tail -20
    exit 1
fi
echo "[INFO] 评测目录: $TRAIN_DIR"

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=${total_tokens:-14336}
export FPS=${FPS:-2}
export datasets="charades-timelens"

RESULTS_FILE="$OPD_OUTPUT/eval_${TAG}_results.txt"
echo "# $TAG Eval ($(date))" > "$RESULTS_FILE"
echo "# TrainDir: $TRAIN_DIR" >> "$RESULTS_FILE"

for step in ${STEPS:-175 150 200 125 100 75 50 25}; do
    tag="${TAG}-step${step}"
    ckpt="$TRAIN_DIR/checkpoint-${step}"
    if [ ! -d "$ckpt" ]; then
        echo "[SKIP] $tag: $ckpt not found"
        echo "[$tag] SKIPPED" >> "$RESULTS_FILE"
        continue
    fi
    echo "=========================================="
    echo "Evaluating: $tag  ($ckpt)"
    echo "=========================================="
    export model_path="$ckpt"
    export pred_path="$OPD_OUTPUT/eval_preds"
    bash scripts/eval_timelens_bench.sh 2>&1 | tee /tmp/eval_${tag}.log
    echo "" >> "$RESULTS_FILE"
    echo "## $tag" >> "$RESULTS_FILE"
    grep -E "IOU 0\.[357]:|mIOU:" /tmp/eval_${tag}.log | grep -v "Warning" | tail -4 >> "$RESULTS_FILE"
    echo "[$tag] Done."
done

echo "=========================================="
echo "All $TAG evaluations complete."
echo "=========================================="
cat "$RESULTS_FILE"
