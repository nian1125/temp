#!/bin/bash
# Teacher 推理 50K 样本（coarse_box, 断点续跑）
set -e
ROOT=/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ
MODEL=/mnt/250T_ceph/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT
DATA=/mnt/250T_ceph/zhenjunchen/data/gemini_refined_data.jsonl
SUBSET=$ROOT/output/filter_v2/subset_50k.jsonl
OUT_DIR=$ROOT/output/teacher_50k

cd $ROOT
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Step 0: 抽 50K
if [ -f "$SUBSET" ] && [ "$(wc -l < $SUBSET)" -ge 50000 ]; then
    echo "[Step 0] $SUBSET 已存在 ($(wc -l < $SUBSET) 条)"
else
    echo "[Step 0] 规则过滤 + 抽 50K..."
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
stop = set('the and with are his her from into out that this was for has its can you but not all one who how did get had been'.split())
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
        if len([w for w in qw if w in test_words])/len(qw) < 0.2: continue
        kept.append(line.strip())
print(f'过滤后: {len(kept)}')
subset = random.sample(kept, min(50000, len(kept)))
with open('${SUBSET}','w') as f:
    for l in subset: f.write(l + '\n')
print(f'抽 {len(subset)} 条')
"
fi

# Step 1: 4 卡 teacher 推理（断点续跑）
mkdir -p $OUT_DIR /tmp/tvg_teacher_cache
echo "[Step 1] 4卡 Teacher 推理 (chunk=4, 断点续跑)"
echo "开始: $(date)"

for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/teacher_inference.py \
      --anno_path "$SUBSET" \
      --model_path "$MODEL" \
      --output_path "$OUT_DIR/teacher_scores" \
      --teacher_cache_dir /tmp/tvg_teacher_cache \
      --teacher_strategy coarse_box \
      --total_tokens 4096 \
      --chunk 4 --index $i &
done
wait

echo "完成: $(date)"
TOTAL=$(cat $OUT_DIR/teacher_scores_*.jsonl | python3 -c "import json,sys;valid=sum(1 for l in sys.stdin if json.loads(l).get('teacher_tiou',-1)>=0);print(valid)")
echo "有效: $TOTAL 条"
