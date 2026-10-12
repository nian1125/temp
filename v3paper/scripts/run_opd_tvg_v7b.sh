#!/usr/bin/env bash
# v7b: 精选 2,493 样本 + JSD 蒸馏 + Teacher Quality Gate
#
# 变更 vs v7a:
#   - teacher_quality_gate: False → True
#
# Gate 机制: 用 teacher argmax 解码估算 teacher tIoU,
# 以 teacher_tiou 加权 JSD loss — teacher 越准, JSD 权重越大。
# 在精选数据(低 IoU 难样本)上, teacher 质量差异可能更大,
# gate 可避免在 teacher 也做错的样本上误导 student。
#
# Usage: bash scripts/run_opd_tvg_v7b.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/data/timelens_2500.jsonl" \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --loss_type bnpo \
    --teacher_quality_gate True \
    --steps_per_generation 1 \
    --num_generations 8 \
    --global_batch_size 64 \
    --max_steps 100 \
    --save_steps 10
