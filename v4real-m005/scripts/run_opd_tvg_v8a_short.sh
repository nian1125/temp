#!/usr/bin/env bash
# v8a-short: Same as v8a but only 200 steps with denser checkpointing
#
# Purpose: Check early checkpoint performance (step 25-200) since v8a's
# best checkpoints were likely in this range but got deleted by save_total_limit=6.
#
# Config identical to v8a: batch=8, spg=8, num_gen=8, lr=1e-6, no gate
# Changes: max_steps=200, save_steps=25 → 8 checkpoints (25,50,...,200)
# Estimated time: ~1.5 hours (vs ~7h for full 1000 steps)
#
# Usage: bash scripts/run_opd_tvg_v8a_short.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 200 \
    --save_steps 25
