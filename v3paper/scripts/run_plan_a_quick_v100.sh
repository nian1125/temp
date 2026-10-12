#!/bin/bash
# 方案A-Quick 全流程: 抽10K → teacher推理 → 过滤 → 2D分桶
# 支持: 断点续跑, 完整记录, screen友好的进度输出
set -e

ROOT=/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ
MODEL=/mnt/250T_ceph/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT
DATA=/mnt/250T_ceph/zhenjunchen/data/gemini_refined_data.jsonl
OUT_DIR=$ROOT/output/teacher_scores_quick
SUBSET=$OUT_DIR/subset_10k.jsonl
SCORES_ALL=$OUT_DIR/teacher_scores_all.jsonl
FILTERED=$OUT_DIR/filtered_10k_gt0.3.jsonl
BENCH_DIR=/mnt/250T_ceph/zhenjunchen/data/TimeLens-Bench

cd $ROOT
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT_DIR output/2d_sets /tmp/tvg_teacher_cache

echo "========================================="
echo " 方案A-Quick: 规则过滤 + teacher推理 + 2D分桶"
echo " $(date)"
echo "========================================="

# ============================================================
# Step 0: 规则过滤 + 抽10K (已有就跳过)
# ============================================================
if [ -f "$SUBSET" ] && [ "$(wc -l < $SUBSET)" -ge 10000 ]; then
    echo "[Step 0] SKIP — $SUBSET 已存在 ($(wc -l < $SUBSET) 条)"
else
    echo "[Step 0] 规则过滤 96K → 抽10K..."
    python scripts/filter_training_data.py
fi

# ============================================================
# Step 1: Teacher 推理 (4卡, 自动跳过已完成的)
# ============================================================
SCORED=0
for f in $OUT_DIR/teacher_scores_*.jsonl; do
    [ -f "$f" ] && SCORED=$((SCORED + $(wc -l < "$f")))
done
echo "[Step 1] 已有 $SCORED 条 teacher 分数"

for i in 0 1 2 3; do
    OUTF=$OUT_DIR/teacher_scores_${i}.jsonl
    DONE=$(wc -l < "$OUTF" 2>/dev/null || echo 0)
    echo "  GPU$i: 已有=$DONE, 目标=~2500"
    CUDA_VISIBLE_DEVICES=$i python -u scripts/teacher_inference.py \
      --anno_path "$SUBSET" \
      --model_path "$MODEL" \
      --output_path $OUT_DIR/teacher_scores \
      --teacher_cache_dir /tmp/tvg_teacher_cache \
      --teacher_strategy coarse_box \
      --chunk 4 --index $i &
done
wait
echo "[Step 1] 推理完成 $(date)"

# 合并所有shard
cat $OUT_DIR/teacher_scores_*.jsonl > $SCORES_ALL
TOTAL=$(wc -l < $SCORES_ALL)
echo "  teacher 分数总计: $TOTAL 条"

# ============================================================
# Step 2: 过滤 tIoU≥0.3 + 统计分布
# ============================================================
echo ""
echo "[Step 2] 数据过滤"

python3 -c "
import json, numpy as np

# 加载 teacher 分数
scores = {}
with open('${SCORES_ALL}') as f:
    for line in f:
        r = json.loads(line)
        scores[r['_index']] = r['teacher_tiou']

# 加载子集标注
annos = []
with open('${SUBSET}') as f:
    for i, line in enumerate(f):
        if line.strip():
            annos.append((i, json.loads(line.strip())))

# 统计 teacher_tIoU 分布
tious = [v for v in scores.values() if v >= 0]
tious.sort()
print(f'Teacher 分数分布 (N={len(tious)}):')
print(f'  mean={np.mean(tious):.4f}  median={np.median(tious):.4f}')
print(f'  ≥0.7: {sum(1 for t in tious if t>=0.7)/len(tious)*100:.1f}%')
print(f'  ≥0.5: {sum(1 for t in tious if t>=0.5)/len(tious)*100:.1f}%')
print(f'  ≥0.3: {sum(1 for t in tious if t>=0.3)/len(tious)*100:.1f}%')
print(f'  <0.1: {sum(1 for t in tious if t<0.1)/len(tious)*100:.1f}%')

# 过滤 tIoU≥0.3
TH=0.3
kept = []
for idx, anno in annos:
    tiou = scores.get(idx, -1)
    if tiou >= TH:
        anno['teacher_tiou'] = round(tiou, 4)
        kept.append(json.dumps(anno, ensure_ascii=False))

with open('${FILTERED}', 'w') as f:
    for l in kept: f.write(l + '\n')
print(f'过滤 tIoU≥{TH}: {len(kept)}/{len(annos)} ({len(kept)/len(annos)*100:.1f}%)')
print(f'输出: ${FILTERED}')
"

# ============================================================
# Step 3: 2D 难度分桶 (需先有 student_iou — 先跳过, 标注用 plain teacher)
# ============================================================
echo ""
echo "[Step 3] 2D 分桶 (暂时用 teacher_tiou 单独标注)"

echo ""
echo "========================================="
echo " 全部完成 $(date)"
echo " 过滤后的训练数据: $FILTERED"
echo " 下一步: 等 student 推理完成后跑 make_2d_difficulty_set.py"
echo "========================================="
