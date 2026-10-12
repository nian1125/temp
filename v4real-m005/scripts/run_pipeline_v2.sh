#!/bin/bash
# 完整数据过滤 pipeline: 拿 student_iou(TimeLens官方) + teacher_iou(coarse_box) → 2D分桶
set -e

ROOT=/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ
MODEL_SFT=/mnt/250T_ceph/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT
DATA=/mnt/250T_ceph/zhenjunchen/data/gemini_refined_data.jsonl
OUT_DIR=$ROOT/output/filter_v2
SUBSET=$OUT_DIR/subset.jsonl
S_SCORES_ALL=$OUT_DIR/student_scores_all.jsonl
T_SCORES_ALL=$OUT_DIR/teacher_scores_all.jsonl
BOTH_IOU=$OUT_DIR/both_iou.jsonl

cd $ROOT
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT_DIR output/2d_sets /tmp/tvg_teacher_cache

echo "========================================="
echo " Pipeline V2: TimeLens官方student + coarse_box teacher → 2D分桶"
echo " $(date)"
echo "========================================="

# ============================================================
# Step 0: 规则过滤 + 随机抽 (用更多条: 5000)
# ============================================================
SAMPLE_N=5000
if [ -f "$SUBSET" ] && [ "$(wc -l < $SUBSET)" -ge $SAMPLE_N ]; then
    echo "[Step 0] 已有 $SUBSET ($(wc -l < $SUBSET) 条)"
else
    echo "[Step 0] 规则过滤(时长30-300s+query重叠) → 抽${SAMPLE_N}..."
    python3 -c "
import json, re, random
random.seed(42)

# 测试集词表
test_words = set()
for name in ['charades-timelens','activitynet-timelens','qvhighlights-timelens']:
    with open('/mnt/250T_ceph/zhenjunchen/data/TimeLens-Bench/'+name+'.json') as f:
        data = json.load(f)
    for v in data.values():
        if isinstance(v,dict) and 'queries' in v:
            for q in v['queries']:
                for w in re.findall(r'[a-z]{2,}', q.lower()):
                    test_words.add(w)
stop = {'the','and','with','are','his','her','from','into','out','that','this','was','for','has','its','can','you','but','not','all','one','who','how','did','get','had','been'}
test_words -= stop

kept = []
with open('${DATA}') as f:
    for line in f:
        d = json.loads(line)
        dur = float(d.get('duration',0))
        query = d.get('query','').lower()
        if not (30 <= dur <= 300): continue
        qw = [w for w in re.findall(r'[a-z]{2,}', query) if w not in stop]
        if not qw: continue
        ov = len([w for w in qw if w in test_words]) / len(qw)
        if ov < 0.2: continue
        kept.append(line.strip())
subset = random.sample(kept, min(${SAMPLE_N}, len(kept)))
with open('${SUBSET}','w') as f:
    for l in subset: f.write(l + '\n')
print(f'规则过滤: {len(kept)}/{96270} → 抽 {len(subset)}')
"
fi

# ============================================================
# Step 1: Student 推理 (TimeLens 官方配置, GPU 0+1)
# ============================================================
echo "[Step 1] Student推理(TimeLens官方配置, GPU0+1)"
for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/student_inference_official.py \
      --anno_path "$SUBSET" \
      --model_path "$MODEL_SFT" \
      --output_path "$OUT_DIR/student_scores" \
      --chunk 2 --index $i &
done

# ============================================================
# Step 2: Teacher 推理 (coarse_box, GPU 2+3)
# ============================================================
echo "[Step 2] Teacher推理(coarse_box, GPU2+3)"
for i in 0 1; do
    ti=$((i + 0))
    CUDA_VISIBLE_DEVICES=$((i + 2)) python -u scripts/teacher_inference.py \
      --anno_path "$SUBSET" \
      --model_path "$MODEL_SFT" \
      --output_path "$OUT_DIR/teacher_scores" \
      --teacher_cache_dir /tmp/tvg_teacher_cache \
      --teacher_strategy coarse_box \
      --chunk 2 --index $ti &
done

echo "等待推理完成..."
wait
echo "推理完成 $(date)"

# 合并
cat $OUT_DIR/student_scores_*.jsonl > $S_SCORES_ALL
cat $OUT_DIR/teacher_scores_*.jsonl > $T_SCORES_ALL
echo "Student: $(wc -l < $S_SCORES_ALL) 条, Teacher: $(wc -l < $T_SCORES_ALL) 条"

# ============================================================
# Step 3: 对齐 student_iou + teacher_iou
# ============================================================
echo "[Step 3] 对齐 both_iou..."
python3 -c "
import json

# 加载 student (按 _orig_idx)
s_iou = {}
with open('${S_SCORES_ALL}') as f:
    for line in f:
        d = json.loads(line); s_iou[d['_orig_idx']] = d['student_iou']

# 加载 teacher (按 _index)
t_iou = {}
with open('${T_SCORES_ALL}') as f:
    for line in f:
        d = json.loads(line); t_iou[d['_index']] = d['teacher_tiou']

# 对齐输出
with open('${SUBSET}') as fin, open('${BOTH_IOU}','w') as fout:
    for i, line in enumerate(fin):
        d = json.loads(line)
        if i in s_iou and i in t_iou:
            d['student_iou'] = round(s_iou[i], 4)
            d['teacher_tiou'] = round(t_iou[i], 4)
            fout.write(json.dumps(d, ensure_ascii=False) + '\n')

print(f'对齐完成: $(wc -l < ${BOTH_IOU}) 条 → ${BOTH_IOU}')
"

# ============================================================
# Step 4: 2D 难度分桶
# ============================================================
echo "[Step 4] 2D分桶..."
python scripts/make_2d_difficulty_set.py --pool "$BOTH_IOU" --out_dir output/2d_sets

echo ""
echo "========================================="
echo " Pipeline 完成 $(date)"
echo " cliff: output/2d_sets/cliff.jsonl"
echo " grpo:  output/2d_sets/grpo.jsonl"
echo "========================================="
