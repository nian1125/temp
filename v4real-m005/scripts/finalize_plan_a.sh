#!/bin/bash
# 补跑 Plan A 的 Step2: 干净聚合 teacher 分片 + 按 tIoU>=0.3 过滤出 66K 训练集。
# 纯 CPU 文件操作, 可在登录节点直接 bash。
set -e
OUTDIR=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores
ANNO=/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens/data/filter-data/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl

echo "=== 分片文件 ==="
ls -la "$OUTDIR"/teacher_scores_[0-7].jsonl
# 干净聚合: 只拼 0-7, 不含 all 自身 (避免 '输入即输出')
cat "$OUTDIR"/teacher_scores_[0-7].jsonl > "$OUTDIR/teacher_scores_all.jsonl"
echo "聚合完成: $(wc -l < "$OUTDIR/teacher_scores_all.jsonl") 条 teacher 分数"

THRESHOLD=0.3
python3 -c "
import json
OUTDIR='$OUTDIR'; ANNO='$ANNO'; threshold=$THRESHOLD
scores={}
with open(f'{OUTDIR}/teacher_scores_all.jsonl') as f:
    for line in f:
        if line.strip():
            r=json.loads(line); scores[r['_index']]=r['teacher_tiou']
annos=[]
with open(ANNO) as f:
    for i,line in enumerate(f):
        if line.strip(): annos.append((i,line.strip()))
kept=[]; dropped=0; no_score=0
for idx,line in annos:
    t=scores.get(idx,-1)
    if t>=threshold: kept.append(line)
    elif t<0: no_score+=1
    else: dropped+=1
fp=f'{OUTDIR}/filtered_teacher_gt{threshold}.jsonl'
with open(fp,'w') as f:
    for line in kept: f.write(line+'\n')
allt=[v for v in scores.values() if v>=0]
print(f'总样本: {len(annos)}')
print(f'保留(tIoU>={threshold}): {len(kept)} ({len(kept)/len(annos)*100:.1f}%)')
print(f'丢弃(毒样本): {dropped}   无分数(报错): {no_score}')
if allt:
    print(f'teacher tIoU: mean={sum(allt)/len(allt):.4f}  >0.5={sum(1 for t in allt if t>0.5)/len(allt)*100:.1f}%  >0.3={sum(1 for t in allt if t>0.3)/len(allt)*100:.1f}%')
print(f'✅ 过滤数据集: {fp}')
"
