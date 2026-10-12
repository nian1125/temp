#!/usr/bin/env bash
# 从零RL实验: base=Qwen3-VL-8B-Instruct(未经SFT) + 393charades数据 + baseline配置(红蓝框+drifting+fixed rescue+师生同prompt) + beta=0.02
# 目的: 验证OPD-GRPO能否不依赖SFT、直接从指令模型起步; 1200步观察长训收敛/平稳区
# 自动链接: 启动后先等待当前训练(beta02)结束, 再开始本实验
# 用法: nohup bash scripts/run_opd_tvg_h20_instruct393_beta02_1200.sh >> /dockerdata/zhenjunchen/logs/run_instruct393.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANNO="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_393_charades_style.jsonl"
MODEL="/dockerdata/zhenjunchen/models/Qwen3-VL-8B-Instruct"

# ============================================================
# 自动链接: 等待当前训练结束 (每60s检查一次)
# ============================================================
if pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; then
    echo ">>> 检测到正在运行的训练, 等待其结束: $(date)"
    while pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; do sleep 60; done
    echo ">>> 当前训练已结束, 本实验启动: $(date)"
fi

# ============================================================
# 模型存在性检查 (fail fast)
# ============================================================
if [ ! -f "${MODEL}/model.safetensors.index.json" ]; then
  echo "FATAL: Instruct模型不存在于 ${MODEL}"
  echo "请先执行: cp -r /mnt/250T_ceph/zhenjunchen/models/Qwen3-VL-8B-Instruct /dockerdata/zhenjunchen/models/"
  exit 1
fi

# ============================================================
# 数据预检 (防静默丢数据): 模拟dataset路径解析, 有缺失则中止
# ============================================================
echo ">>> 数据预检: ${ANNO}"
python3 << 'PYEOF'
import json, os, sys
anno='/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_393_charades_style.jsonl'
missing=0; total=0; badspan=0
with open(anno) as f:
    for line in f:
        d=json.loads(line); total+=1
        vp=d['video_path']
        if not vp.startswith('/'):
            vp=os.path.join('/dockerdata/zhenjunchen/train_videos', vp)
        if not os.path.exists(vp):
            missing+=1
            if missing<=5: print("  MISS:", vp)
        sp=d.get('span',[])
        if not sp or sp[0][0]>=sp[0][1] or sp[0][1]>d.get('duration',1e9)+0.01:
            badspan+=1
print(f"预检: {total}条, 视频缺失={missing}, span异常={badspan}")
if missing>0 or badspan>0:
    print("FATAL: 数据预检未通过, 中止启动")
    sys.exit(1)
print("预检通过 ✅")
PYEOF
if [ $? -ne 0 ]; then exit 1; fi

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

echo "=========================================="
echo "从零RL实验: Qwen3-VL-8B-Instruct + 393 + beta=0.02"
echo "数据: train_reviewed_393_charades_style.jsonl (393条, 已预检)"
echo "配置: 红蓝框 + drifting + fixed rescue + 师生同prompt + beta=0.02"
echo "max_steps=1200, save_steps=50, H20 8卡"
echo "=========================================="

bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --model_path "${MODEL}" \
    --anno_path "${ANNO}" \
    --video_dir "/dockerdata/zhenjunchen/train_videos" \
    --num_devices 8 \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --beta 0.02 \
    --global_batch_size 8 \
    --batch_per_device 1 \
    --steps_per_generation 4 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 1200 \
    --save_steps 50 \
    --teacher_strategy coarse_box_multi \
    --zoom_expand 1.5 \
    --teacher_update_strategy drifting \
    --dead_rescue_enabled True \
    --dead_rescue_mode fixed \
    --dead_reward_std_eps 1e-6 \
    --dead_reward_mean_threshold 0.30 \
    --teacher_advantage_margin 0.05 \
    --dead_rescue_multiplier 2.0 \
    --negative_teacher_multiplier 0.25 \
    --total_tokens 14336 \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2_offload_full.json" \
    --report_to "none" \
    --teacher_prompt_file "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/configs/prompt_student_same.txt" \
    --name_suffix "-instruct393-beta02-1200"

echo "instruct393+beta02 1200步训练完成: $(date)"
