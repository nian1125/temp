#!/usr/bin/env bash
# 方案A-Quick 长训练: 使用过滤后的 6,838 条高质量数据
#
# v8c (方案A-Quick) 在 step175 达到 mIoU=54.47 且仍在上升
# 本次延长到 500 步，save_steps=25 密集保存 20 个 checkpoint
# 覆盖 step 25-500，找到过滤数据的最优训练步数
#
# 数据: /primus_xpfs_root/.../teacher_scores_quick/filtered_10k_gt0.3.jsonl (6,838 条)
# 预计耗时: ~4h (200步 ≈ 1.5h, 500步 ≈ 4h)
#
# 用法: bash scripts/run_plan_a_quick_long.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILTERED_DATA="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl"

if [ ! -f "$FILTERED_DATA" ]; then
    echo "[ERROR] Filtered data not found: $FILTERED_DATA"
    echo "Run scripts/run_plan_a_quick.sh first to generate filtered data."
    exit 1
fi

echo "Using filtered data: $(wc -l < "$FILTERED_DATA") samples from $FILTERED_DATA"

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
