#!/usr/bin/env bash
# 新 H20 机器一键迁移: 模型 + 189训练视频(子目录) + 平铺软链修复 + 校验
# 前提: 新机器已挂载 /mnt/250T_ceph
# 用法: bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/migrate_to_new_h20.sh
set -euo pipefail

CEPH=/mnt/250T_ceph/zhenjunchen
LOCAL=/dockerdata/zhenjunchen
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "H20 新机器迁移: $(date)"
echo "============================================"

# 0. 前置检查
[ -d "${CEPH}/code/Temporal-ZwZ" ] || { echo "FATAL: ceph未挂载 ${CEPH}"; exit 1; }
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

# 3. 训练视频 189个 (子目录布局, cp -n 增量)
echo ">>> 拷贝训练视频 (189个, 增量) ..."
bash "${SCRIPT_DIR}/copy_missing_videos.sh"

# 4. 平铺软链修复 (关键! 393jsonl里71个视频用 train_videos_reviewed/<file> 平铺路径)
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

# 5. 校验: 模拟dataset的路径解析逻辑, 逐条检查393个标注
echo ">>> 校验数据完整性 ..."
python3 << 'PYEOF'
import json, os
missing=0; total=0
with open('/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_393.jsonl') as f:
    for line in f:
        d=json.loads(line); total+=1
        vp=d['video_path']
        if not vp.startswith('/'):
            vp=os.path.join('/dockerdata/zhenjunchen/train_videos', vp)
        if not os.path.exists(vp):
            missing+=1
            if missing<=3: print("  MISS:", vp)
print(f"视频校验: {total-missing}/{total} 条标注可解析 ({total-missing==total and '全部OK' or '有缺失!'})")
PYEOF

# 6. 模型完整性
python3 -c "
import json, os
idx='/dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT/model.safetensors.index.json'
files=set(json.load(open(idx))['weight_map'].values())
miss=[f for f in files if not os.path.exists(os.path.join(os.path.dirname(idx), f))]
print(f'模型校验: {len(files)-len(miss)}/{len(files)} 权重分片存在' + (' (全部OK)' if not miss else f' 缺失{miss}'))
"

echo "============================================"
echo "迁移完成: $(date)"
echo "下一步: bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/setup_h20_venv.sh"
echo "============================================"
