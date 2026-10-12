#!/bin/bash
# 生成"最优OPSD=coarsebox峰值"的Charades预测文件(供badcase蓝条叠加)。GPU job。
# 自动找 coarsebox-sft 训练目录 + 从eval结果挑峰值step; 也可 export STEP=125 手动指定。
set -e
cd /primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
export PYTHONPATH="./:${PYTHONPATH:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

OPD=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo
TRAIN_DIR=$(ls -dt $OPD/opd-*-coarsebox-sft-2026* 2>/dev/null | head -1)
[ -z "$TRAIN_DIR" ] && { echo "❌ 没找到 coarsebox-sft 训练目录"; ls -d $OPD/opd-* | tail; exit 1; }
echo "训练目录: $TRAIN_DIR"

# 峰值step: 优先用 STEP env; 否则从eval结果txt里挑mIOU最大的step
RES=$OPD/eval_coarsebox-sft_results.txt
if [ -z "$STEP" ] && [ -f "$RES" ]; then
  STEP=$(python3 -c "
import re,sys
cur=None;best=(-1,None)
for l in open('$RES'):
    m=re.search(r'step(\d+)',l);
    if m:cur=m.group(1)
    m2=re.search(r'mIOU:\s*([\d.]+)',l)
    if m2 and cur:
        v=float(m2.group(1))
        if v>best[0]:best=(v,cur)
print(best[1] or '')
")
  echo "从结果文件挑到峰值 step=$STEP (mIOU最大)"
fi
[ -z "$STEP" ] && STEP=200 && echo "[WARN] 没拿到峰值step, 退回 STEP=200"
ckpt=$TRAIN_DIR/checkpoint-$STEP
[ ! -d "$ckpt" ] && { echo "❌ ckpt不存在: $ckpt"; ls $TRAIN_DIR | grep checkpoint; exit 1; }
echo "评测ckpt: $ckpt"

export model_path=$ckpt
export datasets=charades-timelens
export min_tokens=64 total_tokens=14336 FPS=2
export pred_path=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/eval-mybest-coarsebox
bash scripts/eval_timelens_bench.sh
echo "✅ 预测文件: $(ls -t $pred_path/*/charades-timelens.jsonl | head -1)"
