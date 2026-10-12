#!/usr/bin/env bash
# Qwen3.5-9B baseline: fixed_all全配置 + 152数据 (复现55.62配置, 仅换Qwen3.5-9B)
# 与 run_opd_tvg_h20_qwen35_2b_baseline_step125.sh 的唯一区别: MODEL = Qwen3.5-9B
# 前置: 1) bash scripts/setup_h20_venv_qwen35.sh  2) Qwen3.5-9B已拷到dockerdata
# 用法: nohup bash scripts/run_opd_tvg_h20_qwen35_9b_baseline_step125.sh >> /dockerdata/zhenjunchen/logs/run_qwen35_9b_baseline.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANNO="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_152.jsonl"
MODEL="/dockerdata/zhenjunchen/models/Qwen3.5-9B"
VENV_PATH="/dockerdata/zhenjunchen/venv-qwen35"

if pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; then
  echo "[ERROR] 检测到已有训练进程在运行:"
  pgrep -af "train_opd_tvg.py"
  exit 1
fi

echo ">>> 模型预检: ${MODEL}"
for f in config.json tokenizer.json; do
  if [ ! -f "${MODEL}/${f}" ]; then
    echo "FATAL: 模型缺少 ${f}, 中止启动 (Qwen3.5-9B是否已拷到dockerdata?)"
    exit 1
  fi
done
if [ ! -f "${MODEL}/model.safetensors" ] && [ ! -f "${MODEL}/model.safetensors.index.json" ]; then
  echo "FATAL: 模型缺少 model.safetensors 或 model.safetensors.index.json, 中止启动"
  exit 1
fi
echo "模型预检通过 ✅"

echo ">>> venv 预检: ${VENV_PATH} (transformers须>=5.13才有qwen3_5)"
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "FATAL: ${VENV_PATH} 不存在, 先运行: bash scripts/setup_h20_venv_qwen35.sh"
  exit 1
fi
source "${VENV_PATH}/bin/activate"
python3 -c "
import transformers
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
assert 'qwen3_5' in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, \
    f'venv的transformers {transformers.__version__} 不支持qwen3_5, 需>=5.13 (运行 setup_h20_venv_qwen35.sh)'
print(f'  venv transformers={transformers.__version__}, qwen3_5支持 ✅')
"

echo ">>> 数据预检: ${ANNO}"
python3 << 'PYEOF'
import json, os, sys
anno='/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_152.jsonl'
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
echo "Qwen3.5-9B baseline (fixed_all+152, step125)"
echo "模型: ${MODEL} (qwen3_5, GatedDeltaNet混合架构)"
echo "venv: ${VENV_PATH} (transformers 5.13.0)"
echo "数据: train_reviewed_152.jsonl (152条, 已预检)"
echo "步数: max_steps=125, 只保留 checkpoint-125"
echo "=========================================="

bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --model_path "${MODEL}" \
    --anno_path "${ANNO}" \
    --video_dir "/dockerdata/zhenjunchen/train_videos" \
    --num_devices 8 \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --divergence jsd \
    --global_batch_size 8 \
    --batch_per_device 1 \
    --steps_per_generation 4 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 125 \
    --save_steps 25 \
    --save_total_limit 1 \
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
    --name_suffix "-qwen35-9b-fixed-all-152"

echo "Qwen3.5-9B baseline (fixed_all+152, step125) 训练完成: $(date)"
