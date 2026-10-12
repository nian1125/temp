#!/usr/bin/env bash
# v7c: 精选 2,493 样本 + GRPO + JSD 蒸馏
#
# 变更 vs v7a:
#   - loss_type: bnpo → grpo (真正的 PPO clipping)
#   - global_batch_size: 64 → 16 (让 grad_accum=2)
#   - steps_per_generation: 1 → 4 (spg=4 > grad_accum=2, GRPO 激活)
#   - num_generations: 8 (保持, 比 v6 的 4 更多, advantage 估计更稳)
#
# GRPO 激活验证: batch=16, 8GPU, bpd=1 → grad_accum=16/8=2, spg=4>2 ✓
#
# 每步唯一样本: (100÷4)×16 = 400 个 prompt
# 涵盖精选池的 400/2493 = 16%
#
# Usage: bash scripts/run_opd_tvg_v7c.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/data/timelens_2500.jsonl" \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --loss_type grpo \
    --steps_per_generation 4 \
    --num_generations 8 \
    --global_batch_size 16 \
    --max_steps 100 \
    --save_steps 10
