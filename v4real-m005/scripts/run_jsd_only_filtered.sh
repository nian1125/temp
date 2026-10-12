#!/usr/bin/env bash
# 纯 JSD 消融: 去掉 GRPO, 只保留 JSD 蒸馏
# 对比 v8c-short (JSD+GRPO, best=54.47)
# 相同数据 (6,838条), 相同 teacher (numberit+red_border)
# grpo_weight=0 → 纯 JSD 蒸馏, 无 RL 信号
#
# 用法: bash scripts/run_jsd_only_filtered.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILTERED_DATA="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl"

if [ ! -f "$FILTERED_DATA" ]; then
    echo "[ERROR] Filtered data not found: $FILTERED_DATA"
    exit 1
fi

echo "JSD-only ablation: no GRPO ($(wc -l < "$FILTERED_DATA") samples)"

bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --anno_path "$FILTERED_DATA" \
    --teacher_strategy numberit+red_border \
    --kl_weight 1.0 \
    --grpo_weight 0.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 200 \
    --save_steps 25
