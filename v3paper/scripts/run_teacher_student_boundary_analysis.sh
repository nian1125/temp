#!/bin/bash
# ============================================================
# 批量推理：SFT plain vs SFT + coarse_box 1.5×
# 产出 per-sample 的 start/end/center/duration 误差
# 用法: bash scripts/run_teacher_student_boundary_analysis.sh
# ============================================================
set -e

source /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/venv/bin/activate
cd /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ

BENCH=/mnt/250T_ceph/zhenjunchen/data/TimelensBench/charades-timelens.json
VIDEO_DIR=/mnt/250T_ceph/zhenjunchen/data/TimelensBench/video_shards/charades/charades
MODEL=/mnt/250T_ceph/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT
CACHE=/tmp/tvg_teacher_eval_cache
NGPU=4

# ============================================================
# 1. SFT plain（无特权）
# ============================================================
echo "========================================="
echo "Step 1: SFT plain (no privilege)"
echo "========================================="
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/eval_teacher_strategies.py \
        --bench_path "$BENCH" \
        --video_dir "$VIDEO_DIR" \
        --model_path "$MODEL" \
        --teacher_strategy plain \
        --teacher_cache_dir "$CACHE" \
        --chunk $NGPU --index $i &
done
wait
python scripts/eval_teacher_strategies.py --aggregate_dir output/teacher_eval/plain
echo "Plain done."

# ============================================================
# 2. SFT + coarse_box 1.5×（有特权）
# ============================================================
echo "========================================="
echo "Step 2: SFT + coarse_box 1.5x (privileged)"
echo "========================================="
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/eval_teacher_strategies.py \
        --bench_path "$BENCH" \
        --video_dir "$VIDEO_DIR" \
        --model_path "$MODEL" \
        --teacher_strategy coarse_box \
        --teacher_cache_dir "$CACHE" \
        --chunk $NGPU --index $i &
done
wait
python scripts/eval_teacher_strategies.py --aggregate_dir output/teacher_eval/coarse_box
echo "Coarse_box done."

# ============================================================
# 3. 边界误差分析
# ============================================================
echo "========================================="
echo "Step 3: Boundary error analysis"
echo "========================================="
python scripts/analyze_boundary_errors.py \
    --plain_dir output/teacher_eval/plain \
    --privileged_dir output/teacher_eval/coarse_box \
    --output_dir output/teacher_eval/analysis

echo "ALL DONE!"
