#!/bin/bash
# TimeLens 官方评测 (Charades)
# 必须在运行机 cd 到 TimeLens 根目录后执行

set -e
TL_ROOT=/mnt/250T_ceph/zhenjunchen/code/TimeLens
MODEL=/mnt/250T_ceph/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT
DATA_ROOT=/mnt/250T_ceph/zhenjunchen/data

echo "========================================="
echo " TimeLens 官方评测 (Charades)"
echo "========================================="

# 1. 装依赖
echo "[1/4] 安装 nncore..."
pip install "$TL_ROOT/../wheels/nncore-0.4.5.tar.gz" 2>&1 | tail -1
pip install opencv-python 2>/dev/null

# 2. 数据路径 (按 TimeLens 约定)
echo "[2/4] 设置数据路径..."
cd "$TL_ROOT"
mkdir -p data/TimeLens-Bench/videos

# charades-timelens.json
ln -sf "$DATA_ROOT/TimeLens-Bench/charades-timelens.json" data/TimeLens-Bench/charades-timelens.json

# activitynet 和 qvh 的 json (如果有的话也链上)
for f in activitynet-timelens qvhighlights-timelens; do
    [ -f "$DATA_ROOT/TimeLens-Bench/${f}.json" ] && \
        ln -sf "$DATA_ROOT/TimeLens-Bench/${f}.json" data/TimeLens-Bench/${f}.json
done

# Charades 视频
ln -sf "$DATA_ROOT/TimelensBench/video_shards/charades/charades" data/TimeLens-Bench/videos/charades

echo "  charades-timelens.json: $([ -f data/TimeLens-Bench/charades-timelens.json ] && echo OK || echo MISSING)"
echo "  videos/charades: $([ -d data/TimeLens-Bench/videos/charades ] && echo OK || echo MISSING)"

# 3. 设置环境变量
echo "[3/4] 设置环境变量..."
export PYTHONPATH="./:$PYTHONPATH"
export model_path="$MODEL"
export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"
export pred_path="$TL_ROOT/logs/official_eval"

echo "  model: $model_path"
echo "  datasets: $datasets"
echo "  pred_path: $pred_path"

# 4. 跑评测
echo "[4/4] 开始评测..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/eval_timelens_bench.sh
echo "========================================="
echo " 评测完成"
echo " 结果: $pred_path/$(basename $model_path)_*/charades-timelens.jsonl"
echo "========================================="
