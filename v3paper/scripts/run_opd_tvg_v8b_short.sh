#!/usr/bin/env bash
# v8b-short: Same as v8b but only 200 steps with denser checkpointing
#
# Purpose: Check early checkpoint performance (step 25-200) for v8b (with gate).
# Config identical to v8b: batch=8, spg=8, num_gen=8, lr=1e-6, teacher_quality_gate=True
# Changes: max_steps=200, save_steps=25 → 8 checkpoints (25,50,...,200)
# Estimated time: ~1.5 hours
#
# Usage: bash scripts/run_opd_tvg_v8b_short.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 200 \
    --save_steps 25 \
    --teacher_quality_gate True
