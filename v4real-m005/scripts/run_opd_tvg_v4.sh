#!/usr/bin/env bash
# v4: GRPO with real gradient contribution
#
# Key changes from v1 (best so far, mIoU 53.10):
#   steps_per_generation: 1 → 4  (ratio ≠ 1, GRPO clipping activates)
#   loss_type:            bnpo → grpo  (per-sample norm, larger GRPO magnitude)
#   max_steps:            100 → 400  (same 100 gen rounds × 4 steps each)
#   save_steps:           50 → 100  (checkpoints at gen rounds 25/50/75/100)
#
# Everything else matches v1: kl_weight=1.0, lr=1e-6, num_gen=8, temp=1.0
#
# Usage: bash scripts/run_opd_tvg_v4.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --kl_weight 1.0 \
    --steps_per_generation 4 \
    --loss_type grpo \
    --num_generations 4 \
    --max_steps 400 \
    --save_steps 100
