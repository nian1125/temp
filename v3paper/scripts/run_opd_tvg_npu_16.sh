#!/usr/bin/env bash
# NPU 16卡 昇腾910B2C OPD-GRPO 训练
# 参考: pierrrewang/RM_system/scripts/train_qwen-3.6-35b/ascend_910B2C/
# 用法: nohup bash scripts/run_opd_tvg_npu_16.sh >> /dockerdata/zhenjunchen/logs/run_npu.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================
# 0. 自动 kill 上次残留的训练进程 (防显存占用)
# ============================================================
pkill -9 -f "train_opd_tvg.py" 2>/dev/null || true
pkill -9 -f "deepspeed" 2>/dev/null || true
pkill -9 -f "torchrun" 2>/dev/null || true
sleep 5
echo ">>> 已清理上次残留进程"

# ============================================================
# 1. NPU 环境 (参考 start_train_classify_0.8b.sh: 不 source set_env.sh)
#    base 环境已配置好 CANN, 只设环境变量即可
# ============================================================
export OMP_NUM_THREADS=4
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# ============================================================
# 2. venv
# ============================================================
source /dockerdata/zhenjunchen/venv_npu/bin/activate

# ============================================================
# 3. NPU 环境变量 (参考 pierrrewang 脚本)
# ============================================================
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth1
export ASCEND_GLOBAL_LOG_LEVEL=3
# 算子异步报错定位用 (会慢, 定位完去掉)
export ASCEND_LAUNCH_BLOCKING=1
export USE_NPU_GDR=1
export GDR_PREWARM=1

echo "=== NPU 状态 ==="
npu-smi info 2>/dev/null | head -20 || true

echo "=========================================="
echo "NPU 16卡 OPD-GRPO 训练启动: $(date)"
echo "=========================================="

# ============================================================
# 4. 训练参数 - Run B 配置, sdpa attention
# ============================================================
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --model_path "/dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT" \
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
    --name_suffix "-npu16"
