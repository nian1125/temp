#!/bin/bash
# 在 ANet-TimeLens + QVH-TimeLens(非饱和benchmark)上评测某模型的峰值附近step。
# 用法: PATTERN="opd-*-zoom-sft-2026*" TAG=zoom STEPS="200 150 250" bash submit_eval_anetqvh.sh
set -e
cd /primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
export PYTHONPATH="./:${PYTHONPATH:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# 前提检查: ANet/QVH bench + 视频是否存在
TL=/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
for d in data/TimeLens-Bench/activitynet-timelens.json data/TimeLens-Bench/qvhighlights-timelens.json data/TimeLens-Bench/videos/activitynet data/TimeLens-Bench/videos/qvhighlights; do
  if [ ! -e "$TL/$d" ]; then echo "❌ 缺少 $TL/$d → ANet/QVH数据没下全, 无法评测。先下载TimeLens-Bench的anet/qvh部分。"; MISS=1; fi
done
[ -n "$MISS" ] && { echo "中止: 数据不全"; exit 1; }
echo "✅ ANet/QVH 数据齐全"

OPD=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo
TRAIN_DIR=$(ls -dt $OPD/$PATTERN 2>/dev/null | head -1)
[ -z "$TRAIN_DIR" ] && { echo "❌ 没找到 $PATTERN"; exit 1; }
echo "评测目录: $TRAIN_DIR"
export datasets="activitynet-timelens,qvhighlights-timelens"
export min_tokens=64 total_tokens=14336 FPS=2
RES=$OPD/eval_anetqvh_${TAG}_results.txt
echo "# ANet/QVH $TAG ($(date)) $TRAIN_DIR" > $RES
for step in ${STEPS:-200 150 250}; do
  ckpt=$TRAIN_DIR/checkpoint-$step
  [ -d "$ckpt" ] || { echo "[skip $step]"; continue; }
  export model_path=$ckpt
  export pred_path=$OPD/eval_preds_anetqvh
  bash scripts/eval_timelens_bench.sh 2>&1 | tee /tmp/anetqvh_${TAG}_${step}.log
  echo "## $TAG step$step" >> $RES
  awk '/Evaluating Dataset:/{d=$NF} /mIOU:/{print "  "d" mIoU="$2}' /tmp/anetqvh_${TAG}_${step}.log >> $RES
done
echo "==== 结果 ===="; cat $RES
