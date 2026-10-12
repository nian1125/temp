#!/usr/bin/env bash
# v8b: Scaled GRPO + Teacher Quality Gate (方案 B: 线上软过滤)
#
# Same as v8a but with teacher_quality_gate=True.
# batch=8 (1 prompt/GPU, OOM-safe). GRPO: spg=8 > grad_accum=1 ✓
#
# Usage: bash scripts/run_opd_tvg_v8b.sh

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
    --teacher_quality_gate True
