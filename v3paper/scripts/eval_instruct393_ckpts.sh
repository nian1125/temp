#!/bin/bash
# ============================================================
# 批量评测 instruct393-beta02-1200 的多个 checkpoint
# 目的: 判断 step1200 相比前面的 ckpt 是更好还是更坏（找 eval 峰值位置）
# 用法: bash scripts/eval_instruct393_ckpts.sh            # 默认测全部 8 个点
#       bash scripts/eval_instruct393_ckpts.sh 750 1200   # 只测指定 step
# ============================================================
set -u

VENV_PATH="/dockerdata/zhenjunchen/venv"
source "${VENV_PATH}/bin/activate"

TIMELENS_ROOT="/mnt/250T_ceph/zhenjunchen/code/TimeLens"
OPD_OUTPUT="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/opd-grpo"
RUN="${OPD_OUTPUT}/opd-coarse_box_multi-kl1.0-gw1.0-grpo-spg4-lr1e-6-instruct393-beta02-1200-20260817-0044"

# 覆盖: 预测峰值区(300-600) + train峰值区(750) + 平台区(900-1200)
STEPS="${@:-300 400 500 600 750 900 1050 1200}"

cd "$TIMELENS_ROOT" || exit 1
export PYTHONPATH="./:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com
# 只装缺失的包，已装的不动（默认 only-if-needed 策略，不会升级 torch）
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

# 评测参数（与之前所有 OPD 评测一致: 干净视频、不带框）
export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

RESULTS_FILE="${OPD_OUTPUT}/eval_instruct393_results.txt"
{
    echo ""
    echo "# instruct393-beta02-1200 Eval ($(date))"
    echo "# Run: $RUN"
    echo "# Steps: $STEPS"
} >> "$RESULTS_FILE"

for step in $STEPS; do
    ckpt="$RUN/checkpoint-${step}"
    if [ ! -d "$ckpt" ]; then
        echo "[SKIP] step$step: checkpoint 不存在"
        continue
    fi

    echo "=========================================="
    echo "Evaluating: instruct393 step=$step ($(date +%H:%M:%S))"
    echo "=========================================="

    export model_path="$ckpt"
    export pred_path="${OPD_OUTPUT}/eval_preds/instruct393/step${step}"
    mkdir -p "$pred_path"

    bash scripts/eval_timelens_bench.sh 2>&1 | tee /tmp/eval_instruct393_step${step}.log || true

    {
        echo ""
        echo "## step$step"
        grep -E "IOU 0\.[357]:|mIOU:|mIoU" /tmp/eval_instruct393_step${step}.log | grep -v "Warning" | tail -4
    } >> "$RESULTS_FILE"
    echo "[step$step] Done."
done

echo ""
echo "=========================================="
echo "全部评测完成，汇总:"
echo "=========================================="
cat "$RESULTS_FILE"
