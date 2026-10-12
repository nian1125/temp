#!/usr/bin/env bash
# 新 H20 机器一键迁移 v2: 模型17G + 189训练视频(152/270/393三集合集) + 平铺软链 + 校验
# 前提: 新机器已挂载 /mnt/250T_ceph
# 用法: bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/migrate_to_new_h20_v2.sh
set -euo pipefail

CEPH=/mnt/250T_ceph/zhenjunchen
LOCAL=/dockerdata/zhenjunchen
SHARDS=${CEPH}/data/TimeLens-100K/video_shards
LIST=${CEPH}/code/Temporal-ZwZ/output/2d_sets/migrate_video_list.txt

echo "============================================"
echo "H20 新机器迁移 v2: $(date)"
echo "============================================"

# 0. 前置检查
[ -d "${CEPH}/code/Temporal-ZwZ" ] || { echo "FATAL: ceph未挂载 ${CEPH}"; exit 1; }
[ -f "${LIST}" ] || { echo "FATAL: 视频清单不存在 ${LIST}"; exit 1; }
df -h /dockerdata | tail -1

# 1. 目录结构
mkdir -p ${LOCAL}/{models,train_videos/{cosmo_cap,didemo,internvid_vtime,queryd,hirest},teacher_cache,logs}

# 2. 模型 (17G, ceph→本地SSD, 约5-10分钟)
if [ -f "${LOCAL}/models/TimeLens-Qwen3-VL-8B-SFT/model.safetensors.index.json" ]; then
    echo ">>> 模型已存在 (skip)"
else
    echo ">>> 拷贝模型 (17G) ..."
    cp -r ${CEPH}/models/TimeLens-Qwen3-VL-8B-SFT ${LOCAL}/models/
fi

# 3. 训练视频 189个 (子目录布局, 增量: 已存在则跳过)
echo ">>> 拷贝训练视频 (189个, 增量) ..."
copied=0; skipped=0; missing=0
while IFS= read -r rel; do
    ds="${rel%%/*}"; bn="${rel##*/}"
    src="${SHARDS}/${ds}/${ds}/${bn}"
    dst="${LOCAL}/train_videos/${ds}/${bn}"
    if [ -f "${dst}" ]; then skipped=$((skipped+1)); continue; fi
    if [ -f "${src}" ]; then
        cp "${src}" "${dst}" && copied=$((copied+1))
    else
        missing=$((missing+1)); echo "  CEPH缺失: ${src}"
    fi
done < "${LIST}"
echo ">>> 视频: 新增=${copied}, 跳过=${skipped}, 缺失=${missing}"
[ "${missing}" -eq 0 ] || { echo "FATAL: 有视频缺失"; exit 1; }

# 4. 平铺软链 (关键! jsonl里平铺路径 train_videos_reviewed/<file> 依赖它)
# 注意: 文件名可能以 - 开头, ln 必须加 -- 分隔符
echo ">>> 创建平铺软链 ..."
ln -sfn -- ${LOCAL}/train_videos ${LOCAL}/train_videos_reviewed
cd ${LOCAL}/train_videos
for sub in cosmo_cap didemo internvid_vtime queryd hirest; do
    for f in ${sub}/*.mp4; do
        [ -f "$f" ] && ln -sf -- "$f" "$(basename -- "$f")"
    done
done
cd - >/dev/null

# 5. 校验: 逐条检查三份数据集的标注能否解析到视频
echo ">>> 校验数据完整性 ..."
python3 << 'PYEOF'
import json, os, sys
SETS='/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets'
ok=True
for jn in ['train_reviewed_152.jsonl','train_reviewed_balanced_270_human_charades.jsonl','train_reviewed_393_charades_style.jsonl']:
    missing=0; total=0
    with open(f'{SETS}/{jn}') as f:
        for line in f:
            d=json.loads(line); total+=1
            vp=d['video_path']
            if not vp.startswith('/'):
                vp=os.path.join('/dockerdata/zhenjunchen/train_videos', vp)
            if not os.path.exists(vp):
                missing+=1
                if missing<=3: print("  MISS:", vp)
    print(f"  {jn}: {total-missing}/{total} 可解析" + (" ✅" if missing==0 else " ❌"))
    ok = ok and missing==0
sys.exit(0 if ok else 1)
PYEOF

# 6. 模型完整性
python3 -c "
import json, os
idx='/dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT/model.safetensors.index.json'
files=set(json.load(open(idx))['weight_map'].values())
miss=[f for f in files if not os.path.exists(os.path.join(os.path.dirname(idx), f))]
print(f'模型校验: {len(files)-len(miss)}/{len(files)} 权重分片存在' + (' ✅' if not miss else f' 缺失{miss}'))
"

echo "============================================"
echo "迁移完成: $(date)"
echo "下一步: bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/setup_h20_venv.sh"
echo "============================================"
