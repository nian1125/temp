#!/bin/bash
# ============================================================
# 总评测: 所有待测 checkpoint (charades-timelens, 与历史评测配置一致)
#   A. 8个新消融 run 的 ckpt-125 (rkl/jsd-only/ema/red-only/random-blue/kl05/kl20/kl30)
#   B. instruct393 的 8 个点 (300/400/500/600/750/900/1050/1200)
# 评测配置与之前所有 OPD 评测一致: 干净视频不带框, FPS=2, min_tokens=64
# 单点失败不阻塞后续; 结果实时写入 RESULTS_FILE
# 用法: nohup bash scripts/eval_all_pending.sh > eval_all_pending.log 2>&1 &
# ============================================================
set -u

VENV_PATH="/dockerdata/zhenjunchen/venv"
source "${VENV_PATH}/bin/activate"

TIMELENS_ROOT="/mnt/250T_ceph/zhenjunchen/code/TimeLens"
OPD_OUTPUT="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/opd-grpo"

cd "$TIMELENS_ROOT" || exit 1
export PYTHONPATH="./:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

RESULTS_FILE="${OPD_OUTPUT}/eval_all_pending_results.txt"
{
    echo ""
    echo "# eval_all_pending ($(date))"
} >> "$RESULTS_FILE"

# 待测清单: tag|run目录名|steps
# 注: rkl/jsd-only/ema/red-only/random-blue 已手动测完(见用户汇总表), 此处只测剩余11个点
TASKS=(
    "kl05|opd-coarse_box_multi-kl0.5-gw1.0-grpo-spg4-lr1e-6-ab152-kl05-20260818-0016|125"
    "kl20|opd-coarse_box_multi-kl2.0-gw1.0-grpo-spg4-lr1e-6-ab152-kl20-20260818-0131|125"
    "kl30|opd-coarse_box_multi-kl3.0-gw1.0-grpo-spg4-lr1e-6-ab152-kl30-20260818-0246|125"
    "instruct393|opd-coarse_box_multi-kl1.0-gw1.0-grpo-spg4-lr1e-6-instruct393-beta02-1200-20260817-0044|300 400 500 600 750 900 1050 1200"
)

total=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r tag run steps <<< "$task"
    for step in $steps; do
        total=$((total+1))
        ckpt="${OPD_OUTPUT}/${run}/checkpoint-${step}"
        if [ ! -d "$ckpt" ]; then
            echo "[SKIP] ${tag}-step${step}: checkpoint 不存在"
            echo "[${tag}-step${step}] SKIPPED" >> "$RESULTS_FILE"
            continue
        fi

        echo "=========================================="
        echo "[${total}] Evaluating: ${tag} step=${step} ($(date +%H:%M:%S))"
        echo "=========================================="

        export model_path="$ckpt"
        export pred_path="${OPD_OUTPUT}/eval_preds/${tag}/step${step}"
        mkdir -p "$pred_path"

        bash scripts/eval_timelens_bench.sh 2>&1 | tee "/tmp/eval_${tag}_step${step}.log" || true

        {
            echo ""
            echo "## ${tag}-step${step}"
            grep -E "IOU 0\.[357]:|mIOU:|mIoU" "/tmp/eval_${tag}_step${step}.log" | grep -v "Warning" | tail -4
        } >> "$RESULTS_FILE"
        echo "[${tag}-step${step}] Done."
    done
done

echo ""
echo "=========================================="
echo "全部评测完成 ($(date)), 汇总:"
echo "=========================================="
cat "$RESULTS_FILE"
