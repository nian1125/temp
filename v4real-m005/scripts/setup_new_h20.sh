#!/usr/bin/env bash
# 新 H20 机器一键初始化脚本
# 从 ceph 拷贝模型、839训练视频、900候选视频到本地 /dockerdata
# 用法: bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/setup_new_h20.sh
set -euo pipefail

echo "=== [1/4] 创建目录 ==="
mkdir -p /dockerdata/zhenjunchen/{models,logs,train_videos,train_videos_900,teacher_cache}

echo ""
echo "=== [2/4] 拷贝模型 (16G, ~2分钟) ==="
cp -r /mnt/250T_ceph/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT /dockerdata/zhenjunchen/models/ &
PID_MODEL=$!

echo ""
echo "=== [3/4] 拷贝 817个训练视频 → train_videos (~5G, ~30秒) ==="
python3 << 'PYEOF'
import json, os, shutil

ANNOS = '/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_839.jsonl'
DST = '/dockerdata/zhenjunchen/train_videos'
os.makedirs(DST, exist_ok=True)

count = 0
with open(ANNOS) as f:
    for line in f:
        d = json.loads(line)
        src = d['video_path']
        dst = os.path.join(DST, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            count += 1
print(f'Copied {count} videos (skipped existing)')
PYEOF

echo ""
echo "=== [4/4] 拷贝 300个候选视频 → train_videos_900 (~1.5G, ~10秒) ==="
bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/copy_videos_900_to_dockerdata.sh

wait $PID_MODEL 2>/dev/null || true

echo ""
echo "=== DONE ==="
echo "Models:   $(ls /dockerdata/zhenjunchen/models/ | grep -i time | wc -l)"
echo "Videos839: $(ls /dockerdata/zhenjunchen/train_videos/ | wc -l)"
echo "Videos900: $(ls /dockerdata/zhenjunchen/train_videos_900/ | wc -l)"
