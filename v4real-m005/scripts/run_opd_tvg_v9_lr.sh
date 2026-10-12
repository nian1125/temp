#!/usr/bin/env bash
# v9-lr: LR ablation — lr=2e-6 with 10-step warmup (Vision-OPD style)
#
# Base: v8a config (JSD + GRPO, no gate)
# Change: lr 1e-6 → 2e-6, add warmup_steps=10
#
# Usage: bash scripts/run_opd_tvg_v9_lr.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 1000 \
    --save_steps 50 \
    --learning_rate 2e-6 \
    --warmup_steps 10
