#!/usr/bin/env bash
# Ablation: 纯 GRPO (kl_weight=0, 无蒸馏) — H20 8卡
# 用法: nohup bash scripts/run_opd_tvg_h20_ablation_pure_grpo.sh >> /dockerdata/zhenjunchen/logs/run_pure_grpo.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; then
  echo "[ERROR] 检测到已有训练进程在运行:"
  pgrep -af "train_opd_tvg.py"
  exit 1
fi

VENV_PATH="/dockerdata/zhenjunchen/venv"
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

echo "=========================================="
echo "Ablation: 纯 GRPO (kl_weight=0, 无蒸馏)"
echo "数据: train_reviewed_393.jsonl"
echo "H20 8卡"
echo "=========================================="

bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --model_path "/dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT" \
    --anno_path "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_393.jsonl" \
    --video_dir "/dockerdata/zhenjunchen/train_videos" \
    --num_devices 8 \
    --kl_weight 0.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --batch_per_device 1 \
    --steps_per_generation 4 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 350 \
    --save_steps 25 \
    --teacher_strategy coarse_box_multi \
    --zoom_expand 1.5 \
    --teacher_update_strategy cached_snapshot \
    --dead_rescue_enabled True \
    --dead_reward_std_eps 1e-6 \
    --dead_reward_mean_threshold 0.30 \
    --teacher_advantage_margin 0.05 \
    --dead_rescue_multiplier 2.0 \
    --negative_teacher_multiplier 0.25 \
    --total_tokens 14336 \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2_offload_full.json" \
    --report_to "none" \
    --teacher_prompt_file "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/configs/prompt_redblue.txt" \
    --name_suffix "-ablation-pure-grpo"

echo "Ablation 纯GRPO 训练完成: $(date)"
