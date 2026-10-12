#!/bin/bash
# 方案 A Quick: 快速验证数据过滤是否有效
#
# 从 96K 训练数据中随机抽 10K, 跑 teacher inference, 过滤, 训练短实验
# 估计总耗时: ~5h (inference ~3h + train ~1.5h)
#
# 用法: bash scripts/run_plan_a_quick.sh
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
PROJECT_ROOT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3"
MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct"
ANNO_PATH="${TIMELENS_ROOT}/data/filter-data/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl"

export PYTHONPATH="${PROJECT_ROOT}:${TIMELENS_ROOT}:${PYTHONPATH:-}"
pip install decord pillow qwen-vl-utils imageio imageio-ffmpeg 2>/dev/null

OUTPUT_DIR="${PROJECT_ROOT}/output/teacher_scores_quick"
mkdir -p "$OUTPUT_DIR"

# ============================================================
# Step 0: 随机抽 10K 子集
# ============================================================
echo "=========================================="
echo "Step 0: Sample 10K subset from 96K"
echo "=========================================="

SUBSET_PATH="${OUTPUT_DIR}/subset_10k.jsonl"

python3 -c "
import random, json

random.seed(42)
with open('${ANNO_PATH}') as f:
    lines = [l.strip() for l in f if l.strip()]
print(f'Total: {len(lines)}')

subset = random.sample(lines, min(10000, len(lines)))
with open('${SUBSET_PATH}', 'w') as f:
    for l in subset:
        f.write(l + '\n')
print(f'Sampled: {len(subset)} -> ${SUBSET_PATH}')
"

# ============================================================
# Step 1: Teacher 推理 (8 GPU 并行)
# ============================================================
echo ""
echo "=========================================="
echo "Step 1: Teacher Inference on 10K subset"
echo "=========================================="

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))}"
CHUNKS=${#GPULIST[@]}
echo "Using ${CHUNKS} GPUs: ${GPULIST[*]}"

cd "$PROJECT_ROOT"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python scripts/teacher_inference.py \
        --anno_path "$SUBSET_PATH" \
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
TOTAL_SCORED=$(wc -l < "${OUTPUT_DIR}/teacher_scores_all.jsonl")
echo "Aggregated results: ${TOTAL_SCORED} samples"

# ============================================================
# Step 2: 数据筛选
# ============================================================
echo ""
echo "=========================================="
echo "Step 2: Filter Training Data (tIoU > 0.3)"
echo "=========================================="

THRESHOLD=0.3

python3 -c "
import json

# Load teacher scores (index -> tiou mapping)
scores = {}
with open('${OUTPUT_DIR}/teacher_scores_all.jsonl') as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            scores[r['_index']] = r['teacher_tiou']

# Load subset annotations
annos = []
with open('${SUBSET_PATH}') as f:
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
filtered_path = '${OUTPUT_DIR}/filtered_10k_gt${THRESHOLD}.jsonl'
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
# Step 3: 用筛选后的数据训练 (v8a-short config)
# ============================================================
echo ""
echo "=========================================="
echo "Step 3: Train with Filtered Data (short)"
echo "=========================================="

FILTERED_ANNO="${OUTPUT_DIR}/filtered_10k_gt${THRESHOLD}.jsonl"

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
    --max_steps 200 \
    --save_steps 25
