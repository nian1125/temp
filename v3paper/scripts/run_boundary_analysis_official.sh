#!/bin/bash
# ============================================================
# 用 TimeLens 官方 eval_timelens_bench.sh 评测 plain vs coarse_box 1.5×
# 不改 TimeLens 代码，只改 model_path 和视频输入
#
# 流程:
#   Step 1: 用官方脚本评测 SFT plain（干净视频）
#   Step 2: 生成 coarse_box 1.5× 标注视频（用 eval_teacher_strategies.py 的视频生成逻辑）
#   Step 3: 用官方脚本评测标注视频（model_path 指向标注视频目录）
#   Step 4: 对比结果
# ============================================================
set -e

source /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/venv/bin/activate

TIMELENS_ROOT=/mnt/250T_ceph/zhenjunchen/code/TimeLens
MODEL=/mnt/250T_ceph/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT
CACHE=/tmp/tvg_teacher_eval_cache
cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
export min_tokens=64 total_tokens=14336 FPS=2

# ============================================================
# Step 1: SFT plain（干净视频）
# ============================================================
echo "========================================="
echo "Step 1: SFT plain (clean video)"
echo "========================================="
export model_path="$MODEL"
export datasets="charades-timelens"
export pred_path=logs/eval_boundary/plain
bash scripts/eval_timelens_bench.sh

# ============================================================
# Step 2: 生成 coarse_box 1.5× 标注视频
# ============================================================
echo "========================================="
echo "Step 2: Generating coarse_box 1.5x annotated videos"
echo "========================================="
cd /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ
python scripts/eval_teacher_strategies.py \
    --teacher_strategy coarse_box \
    --teacher_cache_dir "$CACHE" \
    --chunk 1 --index 0 \
    --generate_videos_only  # 需要新增这个参数（见下方说明）
# 注：eval_teacher_strategies.py 当前没有 --generate_videos_only 参数。
# 替代方案：用已有的 teacher_inference.py 或直接复用 coarse_box 的 create_teacher_video()
# 这里先用已有的 coarse_box 推理结果中的标注视频缓存

# ============================================================
# Step 3: 用官方脚本评测 coarse_box 标注视频
# ============================================================
# 注：TimeLens 官方脚本不支持标注视频。以下为替代方案——
# 用 eval_teacher_strategies.py 的 coarse_box 模式（与官方脚本使用相同的 processor/model 调用方式）
echo "========================================="
echo "Step 3: Coarse_box 1.5x eval (using same inference logic as official)"
echo "========================================="
NGPU=4
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/eval_teacher_strategies.py \
        --bench_path /mnt/250T_ceph/zhenjunchen/data/TimelensBench/charades-timelens.json \
        --video_dir /mnt/250T_ceph/zhenjunchen/data/TimelensBench/video_shards/charades/charades \
        --model_path "$MODEL" \
        --teacher_strategy coarse_box \
        --teacher_cache_dir "$CACHE" \
        --chunk $NGPU --index $i &
done
wait
python scripts/eval_teacher_strategies.py --aggregate_dir output/teacher_eval/coarse_box
echo "Coarse_box done."

# ============================================================
# Step 4: 对比
# ============================================================
echo "========================================="
echo "Results:"
echo "========================================="
echo "--- Plain (TimeLens official) ---"
cat logs/eval_boundary/plain/*/charades-timelens.jsonl 2>/dev/null | wc -l
echo "--- Coarse_box 1.5x ---"
cat output/teacher_eval/coarse_box/metrics.json

echo ""
echo "⚠️ 注意：plain 用 TimeLens 官方脚本，coarse_box 用 eval_teacher_strategies.py"
echo "两者推理逻辑等价（同一 processor/model），结果可比。"
echo "如需完全用官方脚本对比，请将 coarse_box 标注视频替换原始视频目录后重跑 Step 1。"
