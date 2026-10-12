#!/bin/bash
# =============================================================================
# Temporal-ZwZ Training Script
# Temporal Zoom-In Self-Distillation for Video Temporal Grounding
# Based on Vision-OPD's training pipeline
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION — 根据训练机环境修改
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VISION_OPD_ROOT="/cpfs01/user/chenzhaoyu.czy/code/Vision-OPD"

# --- Model ---
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # 可改为 Qwen3-VL-8B
CONFIG_NAME="vopd"

# --- Teacher (OPD) ---
TEACHER_MODEL_SOURCE="legacy"
TEACHER_REGULARIZATION="ema"
TEACHER_UPDATE_RATE=0.05
ALPHA=0.5  # OPD loss weight: α × JSD + (1-α) × GRPO

# --- Marking Strategy ---
# 选项: numberit, red_border, t2sgrid, numberit+red_border, crop+numberit
MARKING_STRATEGY="numberit"

# --- Training ---
TRAIN_BATCH_SIZE=32       # 视频占显存更多，比图片小
PPO_MIMI_BATCH_SIZE=32
ROLLOUT_N=4
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2  # 7B模型用2卡TP
LR=2e-6
MAX_PROMPT_LENGTH=16384   # 视频token更多，需要更大
MAX_RESPONSE_LENGTH=512   # 时间定位回答较短
TRAIN_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$TRAIN_MAX_MODEL_LEN}"
ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
TRAINER_N_GPUS_PER_NODE=8
TRAINER_NNODES="${WORLD_SIZE:-1}"
TRAINER_TOTAL_EPOCHS=1
TRAINER_SAVE_FREQ=200
TRAINER_LOGGER='["console","tensorboard"]'

# --- Data Paths ---
DATA_DIR="/cpfs01/user/chenzhaoyu.czy/data/temporal_zwz"
TASK_TRAIN_FILE="${DATA_DIR}/charades_train.parquet"

# --- Reward ---
CUSTOM_REWARD_FUNCTION="${PROJECT_ROOT}/tools/reward_fn.py"

# --- Output ---
MODEL_NAME=$(basename "$MODEL_PATH")
EXPERIMENT_NAME="Temporal-ZwZ-${MODEL_NAME}-${MARKING_STRATEGY}"
PROJECT_NAME="Temporal-ZwZ"
TRAINER_DEFAULT_LOCAL_DIR="${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
TRAINER_ROLLOUT_DATA_DIR="${PROJECT_ROOT}/rollouts/${EXPERIMENT_NAME}"
mkdir -p "$TRAINER_ROLLOUT_DATA_DIR"

EXTRA_ARGS=("$@")

# =============================================================================
# ENVIRONMENT
# =============================================================================
export PYTHONPATH="$VISION_OPD_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
export USER="${USER:-$(id -un 2>/dev/null || echo root)}"
ulimit -c 0

echo "============================================="
echo "Temporal-ZwZ Training"
echo "============================================="
echo "Model:      $MODEL_PATH"
echo "Strategy:   $MARKING_STRATEGY"
echo "Data:       $TASK_TRAIN_FILE"
echo "Teacher:    $TEACHER_REGULARIZATION (α=$TEACHER_UPDATE_RATE)"
echo "OPD α:      $ALPHA"
echo "GPUs:       ${TRAINER_N_GPUS_PER_NODE} x ${TRAINER_NNODES} nodes"
echo "============================================="

python3 -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    data.train_files="[\"$TASK_TRAIN_FILE\"]" \
    data.val_files="[]" \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation=error \
    data.shuffle=True \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key=images \
    data.video_key=videos \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.dataloader_num_workers=4 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MIMI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$TRAIN_MAX_MODEL_LEN \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=20480 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=$TEACHER_MODEL_SOURCE \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER_REGULARIZATION \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
    actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.use_reward_loop=False \
    custom_reward_function.path="$CUSTOM_REWARD_FUNCTION" \
    trainer.project_name=$PROJECT_NAME \
    trainer.group_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.n_gpus_per_node=$TRAINER_N_GPUS_PER_NODE \
    trainer.nnodes=$TRAINER_NNODES \
    trainer.save_freq=$TRAINER_SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \
    trainer.val_before_train=False \
    trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir="$TRAINER_ROLLOUT_DATA_DIR" \
    "${EXTRA_ARGS[@]}"
