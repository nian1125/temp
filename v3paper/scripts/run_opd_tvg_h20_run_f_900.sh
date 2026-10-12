#!/usr/bin/env bash
# H20 8卡 OPD-GRPO - Run F: 900 候选标注数据, Run B 配置 (spg=4)
# 自动: 训练前降级包+nvtx补丁, 训练后还原
# 用法: nohup bash scripts/run_opd_tvg_h20_run_f_900.sh >> /dockerdata/zhenjunchen/logs/run_f_900.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CUDA_ACC="/usr/local/lib/python3.12/dist-packages/deepspeed/accelerator/cuda_accelerator.py"

# ============================================================
# 退出时还原环境
# ============================================================
restore_base_env() {
    echo ""
    echo ">>> Restoring base environment"
    pip install -i https://mirrors.cloud.tencent.com/pypi/simple transformers==5.3.0 2>&1 | tail -2
    pip install -i https://mirrors.cloud.tencent.com/pypi/simple trl==0.27.0 2>&1 | tail -2
    if [ -f "${CUDA_ACC}.bak" ]; then
        cp "${CUDA_ACC}.bak" "$CUDA_ACC"
        rm -f "${CUDA_ACC}.bak"
    fi
    python3 -c "import transformers,trl;print(f'  Verified: transformers={transformers.__version__}, trl={trl.__version__}')"
    echo ">>> Restore done: $(date)"
}
trap restore_base_env EXIT

# ============================================================
# 训练前: 配置环境
# ============================================================
source /dockerdata/zhenjunchen/venv/bin/activate

echo ">>> Downgrading packages"
pip install -i https://mirrors.cloud.tencent.com/pypi/simple transformers==4.57.1 2>&1 | tail -2
pip install -i https://mirrors.cloud.tencent.com/pypi/simple --no-deps trl==0.19.1 2>&1 | tail -2

echo ">>> Applying nvtx patch"
[ ! -f "${CUDA_ACC}.bak" ] && cp "$CUDA_ACC" "${CUDA_ACC}.bak"
sed -i '/def range_push(self, msg, domain=None, category=None):/a\        return -1  # nvtx compat' "$CUDA_ACC"

python3 -c "import transformers,trl;print(f'  transformers={transformers.__version__}, trl={trl.__version__}')"

# ============================================================
# NCCL / H20 + RoCE bond
# ============================================================
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
echo "Run F: 900候选标注数据, Run B 配置 (spg=4)"
echo "  300 unique videos, 900 queries"
echo "=========================================="

bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --model_path "/dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT" \
    --anno_path "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_900_dockerdata.jsonl" \
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
    --report_to "none" \
    --name_suffix "-900data"
