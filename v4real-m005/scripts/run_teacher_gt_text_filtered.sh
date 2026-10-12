#!/usr/bin/env bash
# Teacher 策略消融: gt_text + 过滤数据 (6,838条)
# 对比 numberit+red_border (v8c best=54.47)
# gt_text teacher mIoU=52.52 (最强 teacher)
#
# 用法: bash scripts/run_teacher_gt_text_filtered.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILTERED_DATA="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl"

if [ ! -f "$FILTERED_DATA" ]; then
    echo "[ERROR] Filtered data not found: $FILTERED_DATA"
    exit 1
fi

echo "Teacher ablation: gt_text + filtered data ($(wc -l < "$FILTERED_DATA") samples)"

bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --anno_path "$FILTERED_DATA" \
    --teacher_strategy gt_text \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 200 \
    --save_steps 25
