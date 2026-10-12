#!/usr/bin/env bash
# v7a: 精选 2,493 样本 + 纯 JSD 蒸馏 (与 v1 对比，仅改数据)
#
# 变更 vs v1:
#   - anno_path: 全量 96K → 精选 2,493 (Gaussian IoU 课程采样)
#   - max_steps: 300 → 100 (与 TimeLens 对齐)
#   - save_steps: 50 → 10 (密集评估找最佳早停)
#
# 其余与 v1 完全一致: kl=1.0, bnpo, spg=1, batch=64, num_gen=8
#
# 预期: 精选数据 + 相同训练量 → 直接回答 "数据质量 vs 数据数量" 问题
#
# Usage: bash scripts/run_opd_tvg_v7a.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/data/timelens_2500.jsonl" \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --loss_type bnpo \
    --steps_per_generation 1 \
    --num_generations 8 \
    --global_batch_size 64 \
    --max_steps 100 \
    --save_steps 10
