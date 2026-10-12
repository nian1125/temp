#!/usr/bin/env bash
# H20 8卡 OPD-GRPO 训练脚本 - 实验4: JSD+GRPO, 冻结teacher只训student (对照exp1的共享权重self-distill)
# 与 exp1 唯一区别: teacher 用独立冻结的初始SFT模型(全程不更新), 其余全部与 exp1 一致
#   (coarse_box + zoom_expand=1.5 + kl_weight=1.0 + grpo_weight=1.0)
# 用法: bash scripts/run_opd_tvg_h20_exp4.sh
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
# 训练参数 - 实验4: JSD+GRPO, 冻结teacher(独立初始SFT模型,全程不更新)只训student
#   与 exp1 唯一差别 = --freeze_teacher True; 其余(coarse_box/1.5/kl=1.0)完全一致。
#   多一份teacher权重 → 用 zero2_offload_full 把优化器offload到CPU省显存。
# ============================================================
bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --model_path "/dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT" \
    --anno_path "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_839_local.jsonl" \
    --video_dir "/dockerdata/zhenjunchen/train_videos" \
    --num_devices 8 \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 350 \
    --save_steps 25 \
    --teacher_strategy coarse_box \
    --zoom_expand 1.5 \
    --freeze_teacher True \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2_offload_full.json" \
    --report_to "none"
