#!/usr/bin/env bash
# NPU 16卡 串行训练: 两个基础模型依次训练 (各350步, 与当前run同配置)
#   Exp1: base=TimeLens-Qwen3-VL-8B-SFT   -> run name 后缀 -npu16-s1-qwen3
#   Exp2: base=TimeLens2-8B-SFT           -> run name 后缀 -npu16-s2-timelens2
# 数据: train_reviewed_152.jsonl (人工修正152条), 与当前 run 一致
#
# 用法 (先手动停掉当前训练!):
#   nohup bash scripts/run_opd_tvg_npu_16_serial.sh >> /dockerdata/zhenjunchen/logs/run_npu_serial.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================
# 0. 启动前检查: 有残留训练进程则拒绝启动 (防止与当前run抢16卡)
# ============================================================
if pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; then
  echo "[ERROR] 检测到已有训练进程在运行, 请先停掉再启动串行脚本:"
  pgrep -af "train_opd_tvg.py"
  exit 1
fi

# ============================================================
# 1. NPU 环境 (同 run_opd_tvg_npu_16.sh)
# ============================================================
export OMP_NUM_THREADS=4
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
source /dockerdata/zhenjunchen/venv_npu/bin/activate

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth1
export ASCEND_GLOBAL_LOG_LEVEL=3
# ASCEND_LAUNCH_BLOCKING=1   # 已确认算子稳定(-2221跑到47步无错), 去掉以提速; 若异常再打开定位
export USE_NPU_GDR=1
export GDR_PREWARM=1

echo "=========================================="
echo "NPU 16卡 串行训练启动: $(date)"
echo "=========================================="

run_one() {
  local MODEL="$1"
  local SUFFIX="$2"
  echo "=========================================="
  echo "[$(date)] Exp 开始: base=$MODEL"
  echo "=========================================="
  bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --model_path "$MODEL" \
    --anno_path "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_152.jsonl" \
    --num_devices 16 \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 16 \
    --batch_per_device 1 \
    --steps_per_generation 4 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 350 \
    --save_steps 25 \
    --teacher_strategy coarse_box \
    --zoom_expand 1.5 \
    --teacher_update_strategy cached_snapshot \
    --dead_rescue_enabled True \
    --dead_reward_std_eps 1e-6 \
    --dead_reward_mean_threshold 0.30 \
    --teacher_advantage_margin 0.05 \
    --dead_rescue_multiplier 2.0 \
    --negative_teacher_multiplier 0.25 \
    --total_tokens 14336 \
    --deepspeed "${PROJECT_ROOT}/scripts/zero_offload_npu.json" \
    --report_to "none" \
    --launcher torchrun \
    --tf32 False \
    --max_rollout_batch 2 \
    --name_suffix "$SUFFIX"
}

# Exp1: TimeLens-Qwen3-VL-8B-SFT (NPU机本地, 当前run已用此模型跑通)
run_one "/dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT" "-npu16-s1-qwen3"

# Exp2: TimeLens2-8B-SFT (NPU机本地, 需先从ceph拷到 dockerdata)
run_one "/dockerdata/zhenjunchen/models/TimeLens2-8B-SFT" "-npu16-s2-timelens2"

echo "=========================================="
echo "[$(date)] 串行训练全部完成"
echo "=========================================="
