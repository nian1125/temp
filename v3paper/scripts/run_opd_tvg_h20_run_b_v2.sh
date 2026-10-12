#!/usr/bin/env bash
# H20 8卡 OPD-GRPO 训练脚本 - Run B: coarse_box 1.5x, 增强版
# 与 Run A 区别: steps_per_generation=4, cached_snapshot teacher, dead-group rescue
# 用法: bash scripts/run_opd_tvg_h20_run_b.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ============================================================
# H20 环境
# ============================================================
VENV_PATH="/dockerdata/zhenjunchen/venv"
source "${VENV_PATH}/bin/activate"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL / H20 + RoCE bond
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ============================================================
# Fail-fast 配置核对
# ============================================================
echo "=========================================="
echo "Run B: coarse_box 1.5x (enhanced)"
echo "=========================================="
echo "The enhanced 1.5x run keeps the learning rate and all"
echo "other unspecified parameters identical to Run A. Its only"
echo "training-mechanism changes are cached snapshot teacher"
echo "targets, four updates per rollout block, and bounded"
echo "dead-group-aware JSD reweighting."
echo "=========================================="

# ============================================================
# 训练参数 - Run B: 基础与 Run A 一致, 新增三个机制
# ============================================================
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --model_path "/dockerdata/zhenjunchen/models/TimeLens2-8B-SFT" \
    --anno_path "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_839_local.jsonl" \
    --video_dir "/dockerdata/zhenjunchen/train_videos" \
    --num_devices 8 \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
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
    --deepspeed "${PROJECT_ROOT}/scripts/zero2_offload_full.json" \
    --report_to "none"
