#!/usr/bin/env bash
# SFT: 用 HI-TVG-393 (train_reviewed_393_charades_style.jsonl) 对 TimeLens-Qwen3-VL-8B-SFT 继续SFT
# 代码: TimeLens官方 train_sft_timelens.py (与产出原SFT模型的同一套代码)
# 数据: filtered_hybrid + raw_anno_path (393 jsonl字段完全兼容, span已clip到duration内)
# 超参: 沿用官方recipe (lr 1e-5 cosine, wd 0.1, warmup 0.03, freeze_vision_tower, ZeRO-3),
#       仅适配小数据: global_batch 128->32, epochs 1->3 (393条x3epoch=36步)
# 环境: 原 venv (/dockerdata/zhenjunchen/venv, transformers 4.57.1, qwen3_vl)
# 用法: nohup bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/run_sft_timelens_h20_393.sh >> /mnt/250T_ceph/zhenjunchen/logs/sft_393.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMELENS_ROOT="/mnt/250T_ceph/zhenjunchen/code/TimeLens"
ANNO="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_393_charades_style.jsonl"
MODEL="/dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT"
VENV_PATH="/dockerdata/zhenjunchen/venv"

if pgrep -f "train_sft_timelens.py|train_opd_tvg.py" >/dev/null 2>&1; then
  echo "[ERROR] 检测到已有训练进程在运行:"
  pgrep -af "train_sft_timelens.py|train_opd_tvg.py"
  exit 1
fi

# --- 模型预检 (dockerdata无则从ceph自动拷) ---
if [ ! -f "${MODEL}/model.safetensors.index.json" ]; then
  echo ">>> dockerdata无模型, 从ceph拷贝 (17G) ..."
  mkdir -p /dockerdata/zhenjunchen/models
  cp -r /mnt/250T_ceph/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT "${MODEL}"
fi
python3 -c "
import json, os
idx='${MODEL}/model.safetensors.index.json'
files=set(json.load(open(idx))['weight_map'].values())
miss=[f for f in files if not os.path.exists(os.path.join('${MODEL}',f))]
assert not miss, f'模型分片缺失: {miss}'
print('模型预检通过 ✅')
"

# --- venv 预检 (SFT用原venv: transformers 4.57.1) ---
[ -f "${VENV_PATH}/bin/activate" ] || {
  echo "FATAL: ${VENV_PATH} 不存在, 先运行: bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/setup_h20_venv.sh"; exit 1; }
source "${VENV_PATH}/bin/activate"
python3 -c "
import transformers, peft, nncore, decord
assert transformers.__version__ == '4.57.1', f'transformers={transformers.__version__} (SFT需4.57.1原venv)'
print(f'venv预检通过 ✅ transformers={transformers.__version__}, decord={decord.__version__}')
"

# --- 撤销PyAV源码补丁(若曾被写入): 根因是393相对路径(已修复为绝对), decord默认路径正常 ---
# 恢复官方vision_process.py, 与RL完全一致的默认配置 (幂等: 无补丁则跳过)
python3 << 'PYEOF'
from qwen_vl_utils import vision_process as vp
src_path = vp.__file__
with open(src_path) as f:
    src = f.read()
if "PATCHED-PYAV-BACKEND" in src:
    src = src.split("# ============ PATCHED-PYAV-BACKEND")[0].rstrip() + "\n"
    with open(src_path, "w") as f:
        f.write(src)
    print(f"已撤销PyAV补丁, 恢复官方decord默认路径: {src_path}")
else:
    print("无补丁残留 ✅ (官方原版)")
PYEOF

# --- 数据预检: 393条逐条解析视频 (绝对路径 train_videos_reviewed/ 依赖平铺软链) ---
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
    print("FATAL: 数据预检未通过, 中止启动"); sys.exit(1)
print("数据预检通过 ✅ (393条)")
PYEOF
if [ $? -ne 0 ]; then exit 1; fi

# --- NCCL (H20 + RoCE bond, 与RL脚本一致) ---
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

# --- 超参 ---
GLOBAL_BATCH=32          # 8卡 x 1/dev x accum 4 (官方128, 小数据缩为32)
EPOCHS=1                 # 只要epoch1模型: 393/32向上取整=13步, ~15分钟 (上次3epoch已证明完整链路)
LR=1e-5                  # 官方recipe
TARGET_SIZE=30000        # 分桶每桶上限3333 >> 393, 不触发采样丢失

cd "${TIMELENS_ROOT}"
export PYTHONPATH="${TIMELENS_ROOT}:${PYTHONPATH:-}"
OUTPUT_DIR="${TIMELENS_ROOT}/output/sft-393-timelens-8b-retry/$(date +%Y%m%d-%H%M)"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "SFT: TimeLens-Qwen3-VL-8B-SFT + HI-TVG-393"
echo "模型: ${MODEL}"
echo "数据: ${ANNO} (393条)"
echo "batch=${GLOBAL_BATCH}, epochs=${EPOCHS}, lr=${LR}"
echo "输出: ${OUTPUT_DIR}"
echo "=========================================="

deepspeed training/train/train_sft_timelens.py \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --tf32 True \
  --gradient_checkpointing True \
  --use_liger False \
  --deepspeed scripts/zero3.json \
  --model_name_or_path "${MODEL}" \
  --model_id "timelens-qwen3vl-8b" \
  --conv_type "chatml" \
  --datasets "filtered_hybrid" \
  --raw_anno_path "${ANNO}" \
  --target_size "${TARGET_SIZE}" \
  --remove_unused_columns False \
  --output_dir "${OUTPUT_DIR}" \
  --min_tokens 64 \
  --total_tokens 14336 \
  --fps 2 \
  --fps_max_frames 448 \
  --min_video_len 5 \
  --max_video_len 500 \
  --max_num_words 200 \
  --freeze_vision_tower True \
  --freeze_llm False \
  --freeze_merger False \
  --learning_rate "${LR}" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit "${EPOCHS}" \
  --dataloader_num_workers 4 \
  --seed 42 \
  --report_to none \
  --run_name "sft-393-timelens-8b" \
  --logging_dir wandb \
  --save_only_model True \
  2>&1 | tee "${OUTPUT_DIR}/training.log"

echo "SFT-393 训练完成: $(date)"
echo "输出模型: ${OUTPUT_DIR}/checkpoint-*"
