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
# ⚠️ 这是 v3 冻结快照 (Temporal-ZwZ-v3), 与主线 Temporal-ZwZ 完全隔离。
#    PROJECT_ROOT 必须指向本快照, 否则会去加载主线的 zwz/ 代码, 隔离失效。
PROJECT_ROOT="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v3"
TIMELENS_ROOT="/mnt/250T_ceph/zhenjunchen/code/TimeLens"

MODEL_PATH="/dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT"
# 标注/视频仍复用主线目录(只读, 不复制数据)
ANNO_PATH="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_839_local.jsonl"
VIDEO_DIR="/dockerdata/zhenjunchen/train_videos"

export TIMELENS_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${TIMELENS_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# pip install decord pillow qwen-vl-utils imageio imageio-ffmpeg 2>/dev/null  # H20: 请在 venv 里提前装好

# ============================================================
# Default hyperparameters
# ============================================================
model_id="qwen3-vl-8b-opd"
num_devices=8
batch_per_device=1
global_batch_size=64
epochs=1
seed=42
zoom_expand="2.0"
resume_ckpt=""

# Video processing
min_tokens=64
total_tokens=14336
fps=2

# GRPO
num_generations=8
temperature=1.0
max_completion_length=512
max_rollout_batch=0   # rollout每批generate序列数; 0=一次全量(H20), >0用于NPU省显存
reward_funcs="tiou"
steps_per_generation=1
loss_type="bnpo"

# OPD-specific
kl_weight=2.0
grpo_weight=1.0
kl_temperature=1.0
jsd_alpha=0.5
distillation_topk=100
teacher_strategy="numberit+red_border"
teacher_cache_dir="/dockerdata/zhenjunchen/teacher_cache"
teacher_quality_gate="False"
scale_rewards="False"   # GRPO advantage 是否除以 std (False=Vision-OPD设置, 当前默认)
name_suffix=""          # 可选: 追加到 run_name 用于区分同配置的消融实验
processor_path=""        # 可选: processor从此路径加载(权重仍从model_path)。Qwen2.5-VL借TimeLens-7B processor补时间编码
kl_weight_end=""        # 可选: kl schedule 终值 (空=constant)
kl_schedule="constant"  # constant | linear (kl_weight -> kl_weight_end)
divergence="jsd"        # jsd | fkl | rkl | tvd | symkl | exopd | rlsd
jsd_clip_max="0.0"      # 每token散度截断上限 (>0启用稳定化), 0=不截断
exopd_lambda="1.25"     # ExOPD 外推因子 (divergence=exopd 时生效)
entropy_gate="off"      # off | soft | hard (按teacher熵给蒸馏token加权)
entropy_tau="0.8"       # entropy_gate=hard 的熵阈值
golden_keep_frac="1.0"  # golden-gating: 每prompt保留轨迹比例 (<1启用), 1.0=关闭
two_stage_explore_frac="0.0"  # two-stage: 前该比例步数探索期不惩罚负adv, 0=关闭
exopd_ref_path=""       # ExOPD ref模型路径(空=用base); 4-bit加载
student_mark_strategy="" # ZwZ式: student输入也加标记 (numberit直喂=B / coarse_box=B1)
beta="0.0"              # ref-KL 系数 (ExOPD 用自带4-bit ref, 不需要父类ref, 保持0)
freeze_teacher="False"  # exp4: 冻结teacher(独立初始SFT模型,全程不更新); False=self-distill(teacher=student本体,exp1)
teacher_update_strategy="drifting"  # drifting(默认, 每步用当前权重) | cached_snapshot(每rollout block快照一次)
dead_rescue_enabled="False"  # dead-group rescue: reward死组时增强JSD
dead_rescue_mode="fixed"        # fixed(默认, shuffle前整组检测+逐样本加权) | legacy(复刻Run B/G)
dead_reward_std_eps="1e-6"
dead_reward_mean_threshold="0.30"
teacher_advantage_margin="0.05"
dead_rescue_multiplier="2.0"
negative_teacher_multiplier="0.25"
ref_anno_path=""              # 参考标注路径(如100K), 构建 coarse_box_multi 蓝框映射. 空=用训练数据自身
teacher_prompt_file=""        # teacher prompt 模板文件路径(含 {query}). 空=用策略默认
ema_decay="0.0"               # EMA teacher 衰减系数(0=禁用, >0=启用). 如 0.995

# ---- 新增开关: 默认值均复刻原行为, 老脚本重跑结果不变 ----
teacher_tiou_mode="argmax"        # argmax(原行为) | generate(teacher自主贪心解码, 每block一次)
teacher_gen_max_new_tokens="48"   # generate 模式最大生成token
teacher_gate_mode="off"           # off(原行为=死代码) | on(启用teacher_advantage门控)
dead_rescue_norm="weighted"       # weighted(原行为, batch=1时恒等变换) | mean(死组真正放大)
jsd_token_focus="off"             # off(原行为) | numeric(硬聚焦数值token) | numeric_soft(软加权)
jsd_numeric_weight="4.0"          # numeric_soft 的数值token相对权重

# ---- 论文 Eq5/Eq6 组级 adaptive controller (默认 off = 原行为) ----
adaptive_controller="off"         # off(原batch级近似) | paper(逐组 λ_g∈{0,1,2})
controller_lambda_no_edge="0.0"   # r̄T ≤ r̄S 时的 λ (论文=0, 完全不蒸馏)
controller_lambda_dead="2.0"      # teacher有优势且全零组的 λ (论文=2)
controller_skip_zero="True"       # λ全0时跳过teacher forward(省算力, 结果不变)
dead_group_criterion="std_zero_lowmean"  # std_zero_lowmean(原行为) | all_zero(论文)
teacher_rollout_mode="sample"     # sample(论文: 同temperature采G条) | greedy(1条贪心)
teacher_rollout_n="0"             # 0 = 用 num_generations
teacher_rollout_batch="0"         # 0 = 一次全量; >0 分批省显存
blue_zoom_expand="-1"             # -1=蓝框沿用zoom_expand(历史) | 1.0=论文(蓝框不扩)

# Training
learning_rate=1e-6
warmup_steps=0
max_steps=300
save_steps=50
save_total_limit=20
deepspeed_config="${PROJECT_ROOT}/scripts/zero2.json"   # 可用 --deepspeed 覆盖 (ExOPD用zero2_offload省显存给ref)
output_root="${PROJECT_ROOT}/output/opd-grpo"
report_to="none"
ddp_timeout=7200   # NCCL集合通信超时(秒); 慢解码/offload下防止watchdog误杀. 视频读取偶发IO抖动, 3600太紧改7200
launcher="deepspeed"   # deepspeed | torchrun (NPU用torchrun)
tf32="True"        # CUDA卡用True, NPU必须False

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
    --grpo_weight)      grpo_weight="$2"; shift 2 ;;
    --kl_temperature)   kl_temperature="$2"; shift 2 ;;
    --jsd_alpha)        jsd_alpha="$2"; shift 2 ;;
    --distillation_topk) distillation_topk="$2"; shift 2 ;;
    --teacher_strategy) teacher_strategy="$2"; shift 2 ;;
    --teacher_quality_gate) teacher_quality_gate="$2"; shift 2 ;;
    --scale_rewards)    scale_rewards="$2"; shift 2 ;;
    --max_rollout_batch) max_rollout_batch="$2"; shift 2 ;;
    --name_suffix)      name_suffix="$2"; shift 2 ;;
    --processor_path)   processor_path="$2"; shift 2 ;;
    --kl_weight_end)    kl_weight_end="$2"; shift 2 ;;
    --kl_schedule)      kl_schedule="$2"; shift 2 ;;
    --divergence)       divergence="$2"; shift 2 ;;
    --jsd_clip_max)     jsd_clip_max="$2"; shift 2 ;;
    --exopd_lambda)     exopd_lambda="$2"; shift 2 ;;
    --entropy_gate)     entropy_gate="$2"; shift 2 ;;
    --entropy_tau)      entropy_tau="$2"; shift 2 ;;
    --golden_keep_frac) golden_keep_frac="$2"; shift 2 ;;
    --two_stage_explore_frac) two_stage_explore_frac="$2"; shift 2 ;;
    --fps)              fps="$2"; shift 2 ;;
    --total_tokens)     total_tokens="$2"; shift 2 ;;
    --min_tokens)       min_tokens="$2"; shift 2 ;;
    --exopd_ref_path)   exopd_ref_path="$2"; shift 2 ;;
    --student_mark_strategy) student_mark_strategy="$2"; shift 2 ;;
    --freeze_teacher)   freeze_teacher="$2"; shift 2 ;;
    --teacher_update_strategy) teacher_update_strategy="$2"; shift 2 ;;
    --dead_rescue_enabled) dead_rescue_enabled="$2"; shift 2 ;;
    --dead_rescue_mode) dead_rescue_mode="$2"; shift 2 ;;
    --dead_reward_std_eps) dead_reward_std_eps="$2"; shift 2 ;;
    --dead_reward_mean_threshold) dead_reward_mean_threshold="$2"; shift 2 ;;
    --teacher_advantage_margin) teacher_advantage_margin="$2"; shift 2 ;;
    --dead_rescue_multiplier) dead_rescue_multiplier="$2"; shift 2 ;;
    --negative_teacher_multiplier) negative_teacher_multiplier="$2"; shift 2 ;;
    --reward_funcs)     reward_funcs="$2"; shift 2 ;;
    --beta)             beta="$2"; shift 2 ;;
    --deepspeed)        deepspeed_config="$2"; shift 2 ;;
    --learning_rate)    learning_rate="$2"; shift 2 ;;
    --warmup_steps)     warmup_steps="$2"; shift 2 ;;
    --steps_per_generation) steps_per_generation="$2"; shift 2 ;;
    --loss_type)        loss_type="$2"; shift 2 ;;
    --save_steps)       save_steps="$2"; shift 2 ;;
    --save_total_limit) save_total_limit="$2"; shift 2 ;;
    --num_generations)  num_generations="$2"; shift 2 ;;
    --ddp_timeout)      ddp_timeout="$2"; shift 2 ;;
    --seed)             seed="$2"; shift 2 ;;
    --report_to)        report_to="$2"; shift 2 ;;
    --zoom_expand)      zoom_expand="$2"; shift 2 ;;
    --ref_anno_path)    ref_anno_path="$2"; shift 2 ;;
    --teacher_prompt_file) teacher_prompt_file="$2"; shift 2 ;;
    --ema_decay)        ema_decay="$2"; shift 2 ;;
    --teacher_tiou_mode)          teacher_tiou_mode="$2"; shift 2 ;;
    --teacher_gen_max_new_tokens) teacher_gen_max_new_tokens="$2"; shift 2 ;;
    --teacher_gate_mode)          teacher_gate_mode="$2"; shift 2 ;;
    --dead_rescue_norm)           dead_rescue_norm="$2"; shift 2 ;;
    --jsd_token_focus)            jsd_token_focus="$2"; shift 2 ;;
    --jsd_numeric_weight)         jsd_numeric_weight="$2"; shift 2 ;;
    --adaptive_controller)        adaptive_controller="$2"; shift 2 ;;
    --controller_lambda_no_edge)  controller_lambda_no_edge="$2"; shift 2 ;;
    --controller_lambda_dead)     controller_lambda_dead="$2"; shift 2 ;;
    --controller_skip_zero)       controller_skip_zero="$2"; shift 2 ;;
    --dead_group_criterion)       dead_group_criterion="$2"; shift 2 ;;
    --teacher_rollout_mode)       teacher_rollout_mode="$2"; shift 2 ;;
    --teacher_rollout_n)          teacher_rollout_n="$2"; shift 2 ;;
    --teacher_rollout_batch)      teacher_rollout_batch="$2"; shift 2 ;;
    --blue_zoom_expand)           blue_zoom_expand="$2"; shift 2 ;;
    --resume_from_checkpoint) resume_ckpt="$2"; shift 2 ;;
    --launcher)         launcher="$2"; shift 2 ;;
    --tf32)             tf32="$2"; shift 2 ;;
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
KL_END_ARG=""
if [ -n "${kl_weight_end}" ]; then KL_END_ARG="--kl_weight_end ${kl_weight_end}"; fi
REF_ARG=""
if [ -n "${exopd_ref_path}" ]; then REF_ARG="--exopd_ref_path ${exopd_ref_path}"; fi
SMARK_ARG=""
if [ -n "${student_mark_strategy}" ]; then SMARK_ARG="--student_mark_strategy ${student_mark_strategy}"; fi
PROC_ARG=""
if [ -n "${processor_path}" ]; then PROC_ARG="--processor_path ${processor_path}"; fi
RESUME_ARG=""
if [ -n "${resume_ckpt}" ]; then RESUME_ARG="--resume_from_checkpoint ${resume_ckpt}"; fi
run_name="opd-${teacher_strategy}-kl${kl_weight}-gw${grpo_weight}-${loss_type}-spg${steps_per_generation}-lr${learning_rate}${name_suffix}-${run_tag}"
output_dir="${output_root}/${run_name}"

mkdir -p "${output_dir}" "${teacher_cache_dir}"

echo "=========================================="
echo "OPD-GRPO Training for TVG"
echo "=========================================="
echo "Model:            ${MODEL_PATH}"
echo "Annotations:      ${ANNO_PATH}"
echo "Videos:           ${VIDEO_DIR}"
echo "Teacher strategy: ${teacher_strategy}"
echo "Freeze teacher:   ${freeze_teacher}"
echo "Teacher update:   ${teacher_update_strategy}"
echo "Dead rescue:      ${dead_rescue_enabled}"
echo "Dead rescue norm: ${dead_rescue_norm}"
echo "Teacher tIoU:     ${teacher_tiou_mode}"
echo "Teacher gate:     ${teacher_gate_mode}"
echo "JSD token focus:  ${jsd_token_focus} (w=${jsd_numeric_weight})"
echo "Controller:       ${adaptive_controller} (λ0=${controller_lambda_no_edge} λdead=${controller_lambda_dead})"
echo "Dead criterion:   ${dead_group_criterion}"
echo "Teacher rollout:  ${teacher_rollout_mode} n=${teacher_rollout_n} batch=${teacher_rollout_batch}"
echo "Blue zoom:        ${blue_zoom_expand}"
echo "KL weight:        ${kl_weight}"
echo "GRPO weight:      ${grpo_weight}"
echo "Steps/generation: ${steps_per_generation}"
echo "Loss type:        ${loss_type}"
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

if [ "${launcher}" = "torchrun" ]; then
  LAUNCH_CMD="torchrun --nproc_per_node ${num_devices} --master_port ${MASTER_PORT:-6008} zwz/train/train_opd_tvg.py"
else
  LAUNCH_CMD="deepspeed --num_gpus ${num_devices} zwz/train/train_opd_tvg.py"
fi

${LAUNCH_CMD} \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --tf32 "${tf32}" \
  --gradient_checkpointing True \
  --deepspeed "${deepspeed_config}" \
  --model_name_or_path "${MODEL_PATH}" \
  --model_id "${model_id}" \
  --anno_path "${ANNO_PATH}" \
  --video_dir "${VIDEO_DIR}" \
  --dataset_name "filtered_hybrid" \
  --kl_weight "${kl_weight}" \
  --kl_schedule "${kl_schedule}" \
  --divergence "${divergence}" \
  --jsd_clip_max "${jsd_clip_max}" \
  --exopd_lambda "${exopd_lambda}" \
  --entropy_gate "${entropy_gate}" \
  --entropy_tau "${entropy_tau}" \
  --golden_keep_frac "${golden_keep_frac}" \
  --two_stage_explore_frac "${two_stage_explore_frac}" \
  ${KL_END_ARG} \
  ${REF_ARG} \
  ${SMARK_ARG} \
  ${PROC_ARG} \
  --grpo_weight "${grpo_weight}" \
  --kl_temperature "${kl_temperature}" \
  --jsd_alpha "${jsd_alpha}" \
  --distillation_topk "${distillation_topk}" \
  --teacher_strategy "${teacher_strategy}" \
  --teacher_cache_dir "${teacher_cache_dir}" \
  --teacher_quality_gate "${teacher_quality_gate}" \
  --freeze_teacher "${freeze_teacher}" \
  --teacher_update_strategy "${teacher_update_strategy}" \
  --dead_rescue_enabled "${dead_rescue_enabled}" \
  --dead_rescue_mode "${dead_rescue_mode}" \
  --dead_reward_std_eps "${dead_reward_std_eps}" \
  --dead_reward_mean_threshold "${dead_reward_mean_threshold}" \
  --teacher_advantage_margin "${teacher_advantage_margin}" \
  --dead_rescue_multiplier "${dead_rescue_multiplier}" \
  --negative_teacher_multiplier "${negative_teacher_multiplier}" \
  --ref_anno_path "${ref_anno_path}" \
  --teacher_prompt_file "${teacher_prompt_file}" \
  --ema_decay "${ema_decay}" \
  --teacher_tiou_mode "${teacher_tiou_mode}" \
  --teacher_gen_max_new_tokens "${teacher_gen_max_new_tokens}" \
  --teacher_gate_mode "${teacher_gate_mode}" \
  --dead_rescue_norm "${dead_rescue_norm}" \
  --jsd_token_focus "${jsd_token_focus}" \
  --jsd_numeric_weight "${jsd_numeric_weight}" \
  --adaptive_controller "${adaptive_controller}" \
  --controller_lambda_no_edge "${controller_lambda_no_edge}" \
  --controller_lambda_dead "${controller_lambda_dead}" \
  --controller_skip_zero "${controller_skip_zero}" \
  --dead_group_criterion "${dead_group_criterion}" \
  --teacher_rollout_mode "${teacher_rollout_mode}" \
  --teacher_rollout_n "${teacher_rollout_n}" \
  --teacher_rollout_batch "${teacher_rollout_batch}" \
  --blue_zoom_expand "${blue_zoom_expand}" \
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
  --warmup_steps "${warmup_steps}" \
  --num_train_epochs "${epochs}" \
  --per_device_train_batch_size "${batch_per_device}" \
  --gradient_accumulation_steps "${grad_accum_steps}" \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "${save_steps}" \
  --save_total_limit "${save_total_limit}" \
  --dataloader_num_workers 4 \
  --log_completions True \
  --use_liger False \
  --use_liger_loss False \
  --reward_funcs "${reward_funcs}" \
  --num_generations "${num_generations}" \
  --steps_per_generation "${steps_per_generation}" \
  --temperature "${temperature}" \
  --max_completion_length "${max_completion_length}" \
  --max_rollout_batch "${max_rollout_batch}" \
  --scale_rewards "${scale_rewards}" \
  --loss_type "${loss_type}" \
  --beta "${beta}" \
  --seed "${seed}" \
  --ddp_timeout "${ddp_timeout}" \
  --report_to "${report_to}" \
  --run_name "${run_name}" \
  --logging_dir wandb \
  --save_only_model True \
  --max_steps "${max_steps}" \
  --zoom_expand "${zoom_expand}" \
  ${RESUME_ARG}

echo "=========================================="
echo "Done. Output: ${output_dir}"
echo "=========================================="
