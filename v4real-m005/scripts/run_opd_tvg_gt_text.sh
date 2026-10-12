#!/usr/bin/env bash
# B3: gt_text teacher strategy training
#
# 用 gt_text (TARGET EVENT 横幅 + 红框) 替代 numberit+red_border 作为 teacher 策略
# gt_text teacher eval: mIoU=52.52 (最优), 比 numberit+red_border 的 51.06 高 1.46
#
# 配置同 v8a-short: batch=8, spg=8, num_gen=8, lr=1e-6, no gate
# max_steps=200, save_steps=25 → 8 checkpoints
#
# 训练机完整路径: /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/run_opd_tvg_gt_text.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --teacher_strategy gt_text \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 200 \
    --save_steps 25
