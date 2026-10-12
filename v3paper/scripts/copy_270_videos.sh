#!/usr/bin/env bash
# 270数据集视频搬运: 47个新增视频 ceph→dockerdata (151条用的71个视频已在dockerdata)
# 用法: bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/copy_270_videos.sh
set -euo pipefail

CEPH=/mnt/250T_ceph/zhenjunchen
LOCAL=/dockerdata/zhenjunchen
ANNO="${CEPH}/code/Temporal-ZwZ/output/2d_sets/train_reviewed_balanced_270_human_charades.jsonl"

echo ">>> 提取需要搬运的视频清单: ${ANNO}"
python3 - << 'PYEOF' > /tmp/copy_270_list.txt
import json
anno='/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_balanced_270_human_charades.jsonl'
seen=set()
with open(anno) as f:
    for line in f:
        vp=json.loads(line)['video_path']
        if not vp.startswith('/') and vp not in seen:
            seen.add(vp); print(vp)
PYEOF
echo ">>> 需搬运: $(wc -l < /tmp/copy_270_list.txt) 个视频"

copied=0; skipped=0; missing=0
while IFS= read -r rel; do
    ds="${rel%%/*}"; bn="${rel##*/}"
    src="${CEPH}/data/TimeLens-100K/video_shards/${ds}/${ds}/${bn}"
    dst_dir="${LOCAL}/train_videos/${ds}"
    dst="${dst_dir}/${bn}"
    mkdir -p "${dst_dir}"
    if [ -f "${dst}" ]; then skipped=$((skipped+1)); continue; fi
    if [ -f "${src}" ]; then
        cp "${src}" "${dst}" && copied=$((copied+1))
    else
        missing=$((missing+1)); echo "  CEPH缺失: ${src}"
    fi
done < /tmp/copy_270_list.txt
echo ">>> 搬运完成: 新增=${copied}, 已存在跳过=${skipped}, ceph缺失=${missing}"
[ "${missing}" -eq 0 ] || { echo "FATAL: 有视频缺失"; exit 1; }

# 预检: 模拟dataset路径解析, 270条全部检查
echo ">>> 数据预检 ..."
python3 << 'PYEOF'
import json, os, sys
anno='/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_balanced_270_human_charades.jsonl'
missing=0; total=0; badspan=0
with open(anno) as f:
    for line in f:
        d=json.loads(line); total+=1
        vp=d['video_path']
        if not vp.startswith('/'):
            vp=os.path.join('/dockerdata/zhenjunchen/train_videos', vp)
        if not os.path.exists(vp):
            missing+=1
            if missing<=5: print("  MISS:", vp)
        sp=d.get('span',[])
        if not sp or sp[0][0]>=sp[0][1] or sp[0][1]>d.get('duration',1e9)+0.01:
            badspan+=1
print(f"预检: {total}条, 视频缺失={missing}, span异常={badspan}")
if missing>0 or badspan>0:
    print("FATAL: 预检未通过"); sys.exit(1)
print("预检通过 ✅")
PYEOF
