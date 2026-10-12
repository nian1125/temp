#!/usr/bin/env bash
# S2 实验: teacher≥0.3 + student<0.7 的去简单数据训练
#
# 2,619 条样本: 去掉 student 已经会的简单样本 (学习差距 +0.19)
# 500 步训练, save_steps=25, 20 个 checkpoint
#
# 用法: bash scripts/run_s2_t03_s07.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILTERED_DATA="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_s2_t03_s07.jsonl"

if [ ! -f "$FILTERED_DATA" ]; then
    echo "[ERROR] S2 filtered data not found: $FILTERED_DATA"
    exit 1
fi

echo "S2 experiment: $(wc -l < "$FILTERED_DATA") samples (t>=0.3, s<0.7)"

bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --anno_path "$FILTERED_DATA" \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 500 \
    --save_steps 25
