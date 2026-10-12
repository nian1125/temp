#!/bin/bash
# 方案 A: Teacher 推理 → 数据筛选 → 训练
#
# Step 1: 对全量训练数据跑 Teacher 推理, 得到每个样本的 teacher_tIoU
# Step 2: 筛选 teacher_tIoU > threshold 的样本, 生成新的 annotation 文件
# Step 3: 用筛选后的数据启动 v8c 训练
#
# 用法: bash scripts/run_plan_a.sh
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
PROJECT_ROOT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3"
MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct"
ANNO_PATH="${TIMELENS_ROOT}/data/filter-data/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl"

export PYTHONPATH="${PROJECT_ROOT}:${TIMELENS_ROOT}:${PYTHONPATH:-}"
pip install decord pillow qwen-vl-utils imageio imageio-ffmpeg 2>/dev/null

# ============================================================
# Step 1: Teacher 推理 (8 GPU 并行)
# ============================================================
echo "=========================================="
echo "Step 1: Teacher Inference on Training Data"
echo "=========================================="

OUTPUT_DIR="${PROJECT_ROOT}/output/teacher_scores"
mkdir -p "$OUTPUT_DIR"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))}"
CHUNKS=${#GPULIST[@]}
echo "Using ${CHUNKS} GPUs: ${GPULIST[*]}"

cd "$PROJECT_ROOT"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python scripts/teacher_inference.py \
        --anno_path "$ANNO_PATH" \
        --video_dir "$TIMELENS_ROOT" \
        --model_path "$MODEL_PATH" \
        --output_path "$OUTPUT_DIR/teacher_scores" \
        --teacher_cache_dir "/tmp/tvg_teacher_cache" \
        --chunk "$CHUNKS" \
        --index "$IDX" &
done

echo "Waiting for all ${CHUNKS} inference processes..."
wait
echo "Teacher inference complete."

# Aggregate all shards
cat "${OUTPUT_DIR}"/teacher_scores_*.jsonl > "${OUTPUT_DIR}/teacher_scores_all.jsonl"
echo "Aggregated results: $(wc -l < "${OUTPUT_DIR}/teacher_scores_all.jsonl") samples"

# ============================================================
# Step 2: 数据筛选
# ============================================================
echo ""
echo "=========================================="
echo "Step 2: Filter Training Data"
echo "=========================================="

THRESHOLD=0.3

python3 -c "
import json

scores = {}
with open('${OUTPUT_DIR}/teacher_scores_all.jsonl') as f:
    for line in f:
        r = json.loads(line)
        scores[r['_index']] = r['teacher_tiou']

# Load original annotations
annos = []
with open('${ANNO_PATH}') as f:
    for i, line in enumerate(f):
        if line.strip():
            annos.append((i, line.strip()))

# Filter
threshold = ${THRESHOLD}
kept = []
dropped = 0
no_score = 0
for idx, line in annos:
    tiou = scores.get(idx, -1)
    if tiou >= threshold:
        kept.append(line)
    elif tiou < 0:
        no_score += 1
    else:
        dropped += 1

# Save filtered annotations
filtered_path = '${OUTPUT_DIR}/filtered_teacher_gt${THRESHOLD}.jsonl'
with open(filtered_path, 'w') as f:
    for line in kept:
        f.write(line + '\n')

# Stats
total = len(annos)
all_tious = [v for v in scores.values() if v >= 0]
print(f'Total samples: {total}')
print(f'Kept (teacher_tIoU >= {threshold}): {len(kept)} ({len(kept)/total*100:.1f}%)')
print(f'Dropped (teacher_tIoU < {threshold}): {dropped} ({dropped/total*100:.1f}%)')
print(f'No score (errors): {no_score}')
if all_tious:
    print(f'Teacher tIoU distribution:')
    print(f'  mean={sum(all_tious)/len(all_tious):.4f}')
    print(f'  >0.5: {sum(1 for t in all_tious if t>0.5)/len(all_tious)*100:.1f}%')
    print(f'  >0.3: {sum(1 for t in all_tious if t>0.3)/len(all_tious)*100:.1f}%')
    print(f'  >0.1: {sum(1 for t in all_tious if t>0.1)/len(all_tious)*100:.1f}%')
    print(f'  <0.1: {sum(1 for t in all_tious if t<0.1)/len(all_tious)*100:.1f}%')
print(f'Filtered file saved to: {filtered_path}')
"

# ============================================================
# Step 3: 用筛选后的数据训练 v8c
# ============================================================
echo ""
echo "=========================================="
echo "Step 3: Train v8c with Filtered Data"
echo "=========================================="

FILTERED_ANNO="${OUTPUT_DIR}/filtered_teacher_gt${THRESHOLD}.jsonl"

if [ ! -f "$FILTERED_ANNO" ]; then
    echo "ERROR: Filtered annotation file not found: $FILTERED_ANNO"
    exit 1
fi

echo "Using filtered data: $(wc -l < "$FILTERED_ANNO") samples"

bash scripts/run_opd_tvg.sh \
    --anno_path "$FILTERED_ANNO" \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 1000 \
    --save_steps 50
