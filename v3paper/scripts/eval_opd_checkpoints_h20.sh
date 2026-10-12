#!/bin/bash
# ============================================================
# H20 批量评测 OPD 训练 checkpoint（基于 eval_by_pattern.sh）
# 用法: bash scripts/eval_opd_checkpoints_h20.sh
# ============================================================
set -e

# ============================================================
# H20 环境
# ============================================================
VENV_PATH="/dockerdata/zhenjunchen/venv"
source "${VENV_PATH}/bin/activate"

TIMELENS_ROOT="/mnt/250T_ceph/zhenjunchen/code/TimeLens"
OPD_OUTPUT="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/opd-grpo"

# 要测的两个 run
RUN1="${OPD_OUTPUT}/opd-coarse_box-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260719-1735"
RUN2="${OPD_OUTPUT}/opd-coarse_box-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260719-2311"

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

# 评测参数
export min_tokens=64
export total_tokens=${total_tokens:-14336}
export FPS=${FPS:-2}
export datasets="charades-timelens"  # 可加 activitynet-timelens qvhighlights-timelens

# ============================================================
# 评测函数
# ============================================================
eval_run() {
    local TRAIN_DIR=$1
    local TAG=$2
    local STEPS=$3

    local RESULTS_FILE="${OPD_OUTPUT}/eval_${TAG}_results.txt"
    echo "# $TAG Eval ($(date))" > "$RESULTS_FILE"
    echo "# TrainDir: $TRAIN_DIR" >> "$RESULTS_FILE"

    for step in $STEPS; do
        local ckpt="$TRAIN_DIR/checkpoint-${step}"
        if [ ! -d "$ckpt" ]; then
            echo "[SKIP] $TAG-step$step: checkpoint not found"
            echo "[$TAG-step$step] SKIPPED" >> "$RESULTS_FILE"
            continue
        fi

        echo "=========================================="
        echo "Evaluating: $TAG  step=$step"
        echo "Checkpoint: $ckpt"
        echo "=========================================="

        export model_path="$ckpt"
        export pred_path="${OPD_OUTPUT}/eval_preds"

        bash scripts/eval_timelens_bench.sh 2>&1 | tee /tmp/eval_${TAG}_step${step}.log

        echo "" >> "$RESULTS_FILE"
        echo "## $TAG-step$step" >> "$RESULTS_FILE"
        grep -E "IOU 0\.[357]:|mIOU:" /tmp/eval_${TAG}_step${step}.log | grep -v "Warning" | tail -4 >> "$RESULTS_FILE"
        echo "[$TAG-step$step] Done."
    done

    echo "=========================================="
    echo "$TAG evaluations complete."
    echo "=========================================="
    cat "$RESULTS_FILE"
}

# ============================================================
# 执行评测
# ============================================================
echo "================================================================"
echo "Run1 (1735) - max 150 steps"
echo "================================================================"
eval_run "$RUN1" "coarsebox15-run1" "25 50 75 100 125 150"

echo ""
echo "================================================================"
echo "Run2 (2311) - max 100 steps"
echo "================================================================"
eval_run "$RUN2" "coarsebox15-run2" "25 50 75 100"

echo ""
echo "================================================================"
echo "全部评测完成！"
echo "结果文件:"
echo "  ${OPD_OUTPUT}/eval_coarsebox15-run1_results.txt"
echo "  ${OPD_OUTPUT}/eval_coarsebox15-run2_results.txt"
echo "================================================================"
