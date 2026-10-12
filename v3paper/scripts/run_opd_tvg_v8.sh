#!/usr/bin/env bash
# v8a: Scaled GRPO — batch=8, spg=8, num_gen=8, max_steps=1000
#
# v6 (best, mIoU=53.12) proved GRPO works. v8 scales up num_gen and spg.
#
# OOM history: batch=32 and batch=16 both OOM'd on A100-80GB.
#   Root cause: long videos (up to 14336 tokens) + multiple prompts per GPU.
#   batch=16 (2 prompts/GPU) passed step 1-3 but OOM'd at step 4 when
#   second generation randomly sampled longer videos.
#
# Fix: batch=8 (1 prompt/GPU, same as v6). Eliminates multi-prompt OOM risk.
# Scale via num_gen=8 (v6 had 4) and spg=8 (v6 had 4):
#   - 8 completions/prompt → better advantage estimation
#   - 8 steps/generation → more GRPO signal per generation cycle
#
# GRPO activation: grad_accum = 8/(1×8) = 1, spg=8 > 1 ✓
# Per generation: 8 prompts × 8 completions = 64 reward samples
# Coverage: 1000/8 × 8 = 1,000 unique prompts (1.0%)
#
# Usage: bash scripts/run_opd_tvg_v8.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 1000 \
    --save_steps 50
