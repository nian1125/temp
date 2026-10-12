#!/bin/bash
# =============================================================================
# ZwZ-style RL Training for Temporal Grounding
#
# Student sees MARKED full video frames during training → RL (tIoU reward)
# At inference, marks are removed → model has internalized temporal perception
#
# Uses EasyR1 framework (ZwZ's training framework, a veRL fork)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZWZ_ROOT="/cpfs01/user/chenzhaoyu.czy/code/Zooming-without-Zooming/EasyR1"

# --- Model ---
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

# --- Data ---
DATA_DIR="/cpfs01/user/chenzhaoyu.czy/data/temporal_zwz"
MARKING_STRATEGY="${MARKING_STRATEGY:-numberit}"
SAFE_NAME="${MARKING_STRATEGY//+/_}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/charades_zwz_${SAFE_NAME}.parquet}"

# --- Training ---
MODEL_SHORT="$(basename $MODEL_PATH)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/ZwZ-${MODEL_SHORT}-${SAFE_NAME}}"

# --- RL Config ---
NUM_GPUS_TRAIN="${NUM_GPUS_TRAIN:-8}"
NUM_GPUS_REWARD="${NUM_GPUS_REWARD:-0}"  # 0 = use function reward, no model
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
MINI_ROLLOUT_BATCH_SIZE="${MINI_ROLLOUT_BATCH_SIZE:-48}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"

export PYTHONPATH="$PROJECT_ROOT:$ZWZ_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================="
echo "ZwZ-style RL Training"
echo "  Model: $MODEL_PATH"
echo "  Strategy: $MARKING_STRATEGY"
echo "  Data: $TRAIN_FILE"
echo "  Output: $OUTPUT_DIR"
echo "============================================="

mkdir -p "$OUTPUT_DIR"

python3 -m verl.trainer.main \
    config="${ZWZ_ROOT}/examples/perception_config.yaml" \
    data.train_files="$TRAIN_FILE" \
    data.val_files="[]" \
    data.mini_rollout_batch_size=$MINI_ROLLOUT_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.image_key=images \
    worker.actor.model.model_path="$MODEL_PATH" \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.3 \
    worker.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    algorithm.disable_kl=True \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path="${PROJECT_ROOT}/tools/zwz_reward.py" \
    trainer.experiment_name="ZwZ-${MODEL_SHORT}-${SAFE_NAME}" \
    trainer.save_checkpoint_path="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=$NUM_GPUS_TRAIN \
    trainer.save_limit=3 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    "$@"
