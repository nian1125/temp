#!/bin/bash
# =============================================================================
# SFT Baseline Training Script (for comparison only, NOT the main method)
#
# NOTE: The main Temporal-ZwZ method uses RL (ZwZ-style or OPD-style), not SFT.
# This script trains a vanilla SFT baseline for comparison in ablation tables.
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VISION_OPD_ROOT="/cpfs01/user/chenzhaoyu.czy/code/Vision-OPD"

# --- Model ---
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

# --- Data ---
DATA_DIR="/cpfs01/user/chenzhaoyu.czy/data/temporal_zwz"
TRAIN_FILE="${DATA_DIR}/charades_train_sft.jsonl"

# --- Training ---
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/SFT-$(basename $MODEL_PATH)"
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=8
LR=2e-5
MAX_SEQ_LEN=4096
NUM_FRAMES=32
FPS=2.0

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "============================================="
echo "SFT Baseline Training"
echo "  Model: $MODEL_PATH"
echo "  Data: $TRAIN_FILE"
echo "  Output: $OUTPUT_DIR"
echo "============================================="

torchrun --nproc_per_node=8 \
    "${PROJECT_ROOT}/scripts/train_sft.py" \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs $NUM_EPOCHS \
    --per_device_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --max_seq_len $MAX_SEQ_LEN \
    --num_frames $NUM_FRAMES \
    --fps $FPS \
    --warmup_ratio 0.03 \
    --bf16 \
    --deepspeed "${PROJECT_ROOT}/configs/ds_zero2.json" \
    "$@"
