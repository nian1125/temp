#!/usr/bin/env bash
# v6: Fix GRPO by making ratio ≠ 1.0
#
# Root cause from v4: spg(4) ≤ grad_accum(8), so model weights never change
# within a generation window → ratio = exp(0) = 1.0 exactly → GRPO loss = 0.
#
# Fix: global_batch_size=8 → grad_accum = 8/(1×8GPU) = 1.
# Now spg(4) > grad_accum(1), so the model updates EVERY step:
#   Step 0: generate + train, ratio=1.0 (first step)
#   Step 1: ratio ≠ 1.0 (model updated at step 0) ← GRPO clipping activates
#   Step 2: ratio deviates more
#   Step 3: ratio largest → strongest GRPO signal
#
# Trade-off: effective batch per update = 8 (was 64), noisier gradients.
# But this is how GRPO is designed: small batch + multi-step on-policy updates.
#
# What to monitor:
#   - clip_ratio > 0 (THE key indicator — should be non-zero at steps 1-3)
#   - grpo_loss has actual values (not ~1e-8)
#   - reward trend: should show clearer upward trend than v4
#
# Usage: bash scripts/run_opd_tvg_v6.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 4 \
    --loss_type grpo \
    --num_generations 4 \
    --max_steps 400 \
    --save_steps 100
