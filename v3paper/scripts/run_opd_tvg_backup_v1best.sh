#!/usr/bin/env bash
# OPD-GRPO Training for Temporal Video Grounding
#
# Teacher sees GT-marked video (numberit+red_border), student sees clean video.
# GRPO loss + JSD top-k distillation loss (Vision-OPD style).
#
# Usage:
#   bash scripts/run_opd_tvg.sh
#   bash scripts/run_opd_tvg.sh --kl_weight 0.5 --num_devices 4

set -euo pipefail

# ============================================================
# Paths — adjust for your machine
# ============================================================
TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
PROJECT_ROOT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3"

MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct"
ANNO_PATH="${TIMELENS_ROOT}/data/filter-data/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl"
VIDEO_DIR="${TIMELENS_ROOT}"

export TIMELENS_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${TIMELENS_ROOT}:${PYTHONPATH:-}"

pip install decord pillow qwen-vl-utils imageio imageio-ffmpeg 2>/dev/null

# ============================================================
# Default hyperparameters
# ============================================================
model_id="qwen3-vl-8b-opd"
num_devices=8
batch_per_device=1
global_batch_size=64
epochs=1
seed=42

# Video processing
min_tokens=64
total_tokens=14336
fps=2

# GRPO
num_generations=8
temperature=1.0
max_completion_length=512
reward_funcs="tiou"

# OPD-specific
kl_weight=2.0
kl_temperature=1.0
jsd_alpha=0.5
distillation_topk=100
teacher_strategy="numberit+red_border"
teacher_cache_dir="/tmp/tvg_teacher_cache"

# Training
learning_rate=1e-6
max_steps=300
save_steps=50
deepspeed_config="${PROJECT_ROOT}/scripts/zero1.json"
output_root="${PROJECT_ROOT}/output/opd-grpo"
report_to="none"

# ============================================================
# CLI overrides
# ============================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)       MODEL_PATH="$2"; shift 2 ;;
    --anno_path)        ANNO_PATH="$2"; shift 2 ;;
    --video_dir)        VIDEO_DIR="$2"; shift 2 ;;
    --num_devices)      num_devices="$2"; shift 2 ;;
    --batch_per_device) batch_per_device="$2"; shift 2 ;;
    --global_batch_size) global_batch_size="$2"; shift 2 ;;
    --epochs)           epochs="$2"; shift 2 ;;
    --max_steps)        max_steps="$2"; shift 2 ;;
    --kl_weight)        kl_weight="$2"; shift 2 ;;
    --kl_temperature)   kl_temperature="$2"; shift 2 ;;
    --jsd_alpha)        jsd_alpha="$2"; shift 2 ;;
    --distillation_topk) distillation_topk="$2"; shift 2 ;;
    --teacher_strategy) teacher_strategy="$2"; shift 2 ;;
    --learning_rate)    learning_rate="$2"; shift 2 ;;
    --seed)             seed="$2"; shift 2 ;;
    --report_to)        report_to="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ============================================================
# Derived values
# ============================================================
grad_accum_steps=$((global_batch_size / (batch_per_device * num_devices)))
fps_max_frames=$((total_tokens / min_tokens * 2))
run_tag="$(date +%Y%m%d-%H%M)"
run_name="opd-${teacher_strategy}-kl${kl_weight}-lr${learning_rate}-${run_tag}"
output_dir="${output_root}/${run_name}"

mkdir -p "${output_dir}" "${teacher_cache_dir}"

echo "=========================================="
echo "OPD-GRPO Training for TVG"
echo "=========================================="
echo "Model:            ${MODEL_PATH}"
echo "Annotations:      ${ANNO_PATH}"
echo "Videos:           ${VIDEO_DIR}"
echo "Teacher strategy: ${teacher_strategy}"
echo "KL weight:        ${kl_weight}"
echo "JSD alpha:        ${jsd_alpha}"
echo "Distill top-k:    ${distillation_topk}"
echo "Devices:          ${num_devices}"
echo "Batch:            ${global_batch_size} (${batch_per_device}/dev × ${grad_accum_steps} accum)"
echo "Output:           ${output_dir}"
echo "=========================================="

# ============================================================
# Launch
# ============================================================
cd "${PROJECT_ROOT}"

deepspeed --num_gpus "${num_devices}" zwz/train/train_opd_tvg.py \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --tf32 True \
  --gradient_checkpointing True \
  --deepspeed "${deepspeed_config}" \
  --model_name_or_path "${MODEL_PATH}" \
  --model_id "${model_id}" \
  --anno_path "${ANNO_PATH}" \
  --video_dir "${VIDEO_DIR}" \
  --dataset_name "filtered_hybrid" \
  --kl_weight "${kl_weight}" \
  --kl_temperature "${kl_temperature}" \
  --jsd_alpha "${jsd_alpha}" \
  --distillation_topk "${distillation_topk}" \
  --teacher_strategy "${teacher_strategy}" \
  --teacher_cache_dir "${teacher_cache_dir}" \
  --remove_unused_columns False \
  --output_dir "${output_dir}" \
  --min_tokens "${min_tokens}" \
  --total_tokens "${total_tokens}" \
  --fps "${fps}" \
  --fps_max_frames "${fps_max_frames}" \
  --min_video_len 5 \
  --max_video_len 500 \
  --freeze_vision_tower True \
  --freeze_llm False \
  --freeze_merger False \
  --lr_scheduler_type constant \
  --learning_rate "${learning_rate}" \
  --num_train_epochs "${epochs}" \
  --per_device_train_batch_size "${batch_per_device}" \
  --gradient_accumulation_steps "${grad_accum_steps}" \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "${save_steps}" \
  --save_total_limit 6 \
  --dataloader_num_workers 4 \
  --log_completions True \
  --use_liger False \
  --use_liger_loss False \
  --reward_funcs "${reward_funcs}" \
  --num_generations "${num_generations}" \
  --steps_per_generation 1 \
  --temperature "${temperature}" \
  --max_completion_length "${max_completion_length}" \
  --scale_rewards False \
  --loss_type bnpo \
  --beta 0.0 \
  --seed "${seed}" \
  --report_to "${report_to}" \
  --run_name "${run_name}" \
  --logging_dir wandb \
  --save_only_model True \
  --max_steps "${max_steps}"

echo "=========================================="
echo "Done. Output: ${output_dir}"
echo "=========================================="
