#!/usr/bin/env bash
# SFT smoke: 验证SFT流程能跑通 (同正式配置, 但只跑2步, 不存档)
# 与 run_sft_timelens_h20_393.sh 的区别: max_steps限制(覆盖epochs), save_total_limit 0, 输出目录加-smoke
# 用法: bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/run_sft_timelens_h20_393_smoke.sh 2>&1 | tee /mnt/250T_ceph/zhenjunchen/logs/sft_393_smoke.log
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

[ -f "${MODEL}/model.safetensors.index.json" ] || {
  echo ">>> dockerdata无模型, 从ceph拷贝 (17G) ..."
  mkdir -p /dockerdata/zhenjunchen/models
  cp -r /mnt/250T_ceph/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT "${MODEL}"
}
[ -f "${VENV_PATH}/bin/activate" ] || {
  echo "FATAL: venv不存在, 先跑 setup_h20_venv.sh"; exit 1; }
source "${VENV_PATH}/bin/activate"
python3 -c "
import transformers, decord
assert transformers.__version__ == '4.57.1'
print(f'venv ✅ transformers={transformers.__version__}, decord={decord.__version__}')
"

# 撤销PyAV源码补丁(若上次smoke已写入): 根因是393相对路径(已修复), decord默认路径正常
# 恢复官方vision_process.py, 保持与RL一致的默认配置
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

python3 << 'PYEOF'
import json, os, sys
anno='/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_393_charades_style.jsonl'
missing=0; total=0
with open(anno) as f:
    for line in f:
        d=json.loads(line); total+=1
        vp=d['video_path']
        if not vp.startswith('/'):
            vp=os.path.join('/dockerdata/zhenjunchen/train_videos', vp)
        if not os.path.exists(vp):
            missing+=1
            if missing<=5: print("  MISS:", vp)
print(f"预检: {total}条, 视频缺失={missing}")
if missing>0:
    print("FATAL: 有视频缺失, 中止"); sys.exit(1)
print("数据预检通过 ✅ (393条)")
PYEOF
if [ $? -ne 0 ]; then exit 1; fi

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

cd "${TIMELENS_ROOT}"
export PYTHONPATH="${TIMELENS_ROOT}:${PYTHONPATH:-}"
OUTPUT_DIR="${TIMELENS_ROOT}/output/sft-393-timelens-8b-smoke/$(date +%Y%m%d-%H%M)"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "SFT SMOKE: TimeLens-8B-SFT + HI-TVG-393, 只跑2步"
echo "验证: 数据加载(decord读视频) -> forward/backward -> ZeRO-3"
echo "预期: ~5-8分钟"
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
  --target_size 30000 \
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
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs 1 \
  --max_steps 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --save_strategy no \
  --save_total_limit 0 \
  --dataloader_num_workers 4 \
  --seed 42 \
  --report_to none \
  --run_name "sft-393-smoke" \
  --logging_dir wandb \
  --save_only_model True \
  2>&1 | tee "${OUTPUT_DIR}/training.log"

echo "=========================================="
echo "SFT SMOKE 通过 ✅: $(date)"
echo "正式SFT: nohup bash scripts/run_sft_timelens_h20_393.sh >> logs/sft_393.log 2>&1 &"
echo "=========================================="
