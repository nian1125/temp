#!/usr/bin/env bash
# S3 实验: teacher≥0.5 + student<0.7 的高价值数据训练
#
# 1,815 条样本: teacher 确信做对, student 还不会 (学习差距 +0.27)
# 500 步训练, save_steps=25, 20 个 checkpoint
#
# 用法: bash scripts/run_s3_t05_s07.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILTERED_DATA="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_s3_t05_s07.jsonl"

if [ ! -f "$FILTERED_DATA" ]; then
    echo "[ERROR] S3 filtered data not found: $FILTERED_DATA"
    exit 1
fi

echo "S3 experiment: $(wc -l < "$FILTERED_DATA") samples (t>=0.5, s<0.7)"

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
