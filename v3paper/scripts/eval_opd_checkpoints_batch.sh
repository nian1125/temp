#!/bin/bash
# 批量评测 OPD 训练 checkpoint
# 用法: bash eval_opd_checkpoints_batch.sh
set -e

# ============================================================
# 配置
# ============================================================
TIMELENS_ROOT="/mnt/250T_ceph/zhenjunchen/code/TimeLens"
OPD_OUTPUT="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/opd-grpo"

# 两个训练 run
RUN1="${OPD_OUTPUT}/opd-coarse_box-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260719-1735"
RUN2="${OPD_OUTPUT}/opd-coarse_box-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260719-2311"

# 要测的 checkpoint 列表
RUN1_STEPS="25 50 75 100 125 150"
RUN2_STEPS="25 50 75 100"

# 评测参数（与官方一致）
MIN_TOKENS=64
TOTAL_TOKENS=14336
FPS=2

# 测试集（三选一或全测）
DATASETS="charades-timelens"  # 可改为 "charades-timelens activitynet-timelens qvhighlights-timelens"

# ============================================================
# 环境准备
# ============================================================
cd "${TIMELENS_ROOT}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com

pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null

# ============================================================
# 评测函数
# ============================================================
eval_checkpoint() {
    local ckpt_path=$1
    local tag=$2
    local dataset=$3

    local ckpt_name=$(basename "$ckpt_path")
    local pred_dir="output/eval-${tag}/${dataset}"
    local result_file="${pred_dir}/${ckpt_name}/results.txt"

    echo "============================================="
    echo "评测: ${tag} / ${ckpt_name} / ${dataset}"
    echo "路径: ${ckpt_path}"
    echo "============================================="

    # 跳过已完成的
    if [ -f "$result_file" ] && grep -q "mIoU" "$result_file" 2>/dev/null; then
        echo "⏭ 已完成，跳过"
        return 0
    fi

    mkdir -p "${pred_dir}/${ckpt_name}"

    export model_path="$ckpt_path"
    export datasets="$dataset"
    export min_tokens=$MIN_TOKENS
    export total_tokens=$TOTAL_TOKENS
    export FPS=$FPS
    export pred_path="${pred_dir}/${ckpt_name}"

    bash scripts/eval_timelens_bench.sh 2>&1 | tee "${pred_dir}/${ckpt_name}/eval.log"

    echo "✅ 完成: ${tag} / ${ckpt_name} / ${dataset}"
    echo ""
}

# ============================================================
# 批量评测
# ============================================================
for dataset in $DATASETS; do
    echo "======================================================"
    echo "数据集: $dataset"
    echo "======================================================"

    # Run1
    for step in $RUN1_STEPS; do
        ckpt="${RUN1}/checkpoint-${step}"
        if [ -d "$ckpt" ]; then
            eval_checkpoint "$ckpt" "run1-1735" "$dataset"
        else
            echo "⚠ checkpoint 不存在: $ckpt"
        fi
    done

    # Run2
    for step in $RUN2_STEPS; do
        ckpt="${RUN2}/checkpoint-${step}"
        if [ -d "$ckpt" ]; then
            eval_checkpoint "$ckpt" "run2-2311" "$dataset"
        else
            echo "⚠ checkpoint 不存在: $ckpt"
        fi
    done
done

# ============================================================
# 汇总结果
# ============================================================
echo ""
echo "======================================================"
echo "汇总结果"
echo "======================================================"
echo ""

for dataset in $DATASETS; do
    echo "--- $dataset ---"
    printf "%-20s %-15s %s\n" "Run" "Checkpoint" "mIoU"
    printf "%-20s %-15s %s\n" "---" "----------" "----"

    for tag in run1-1735 run2-2311; do
        pred_dir="output/eval-${tag}/${dataset}"
        if [ -d "$pred_dir" ]; then
            for ckpt_dir in "$pred_dir"/*/; do
                [ -d "$ckpt_dir" ] || continue
                step=$(basename "$ckpt_dir")
                result=$(grep -o "mIoU[^ ]* [0-9.]*" "${ckpt_dir}/results.txt" 2>/dev/null | head -1)
                [ -z "$result" ] && result="未找到"
                printf "%-20s %-15s %s\n" "$tag" "step-$step" "$result"
            done
        fi
    done
    echo ""
done

echo "全部评测完成！"
