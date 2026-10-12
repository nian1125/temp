#!/usr/bin/env bash
# 新H20机器验收: 在训练启动前验证 视频完整性 + venv内decord训练同路径冒烟
# 背景(2026-09-13教训): 某台H20的decord库多进程seek损坏, 单帧冒烟测不出,
#   训练时才崩 (SFT卡了两天)。本脚本模拟训练真实读法: venv内 + qwen_vl_utils decord后端
#   + fps=2采样中间帧 + 8进程并发(模拟DataLoader worker)。
# 用法: bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/verify_new_h20.sh
set -euo pipefail

CEPH=/mnt/250T_ceph/zhenjunchen
LOCAL=/dockerdata/zhenjunchen
MD5LIST=${CEPH}/code/Temporal-ZwZ/output/2d_sets/migrate_video_md5.txt

echo "===== [1/2] 视频md5校验 (189个) ====="
[ -f "${MD5LIST}" ] || { echo "FATAL: md5清单不存在 ${MD5LIST}"; exit 1; }
ok=0; bad=0
while read -r expected rel; do
    ds="${rel%%/*}"; bn="${rel##*/}"
    dst="${LOCAL}/train_videos/${ds}/${bn}"
    if [ -f "${dst}" ]; then
        actual=$(md5sum "${dst}" | awk '{print $1}')
    else
        actual="MISSING"
    fi
    if [ "${actual}" = "${expected}" ]; then
        ok=$((ok+1))
    else
        bad=$((bad+1)); echo "  坏: ${rel} (期望${expected:0:8} 实际${actual:0:8})"
    fi
done < "${MD5LIST}"
echo ">>> md5: 完好=${ok}, 坏=${bad}"
if [ "${bad}" -gt 0 ]; then
    echo "FATAL: 有视频损坏, 重拷这些文件后再验收 (cp ceph源到对应dockerdata路径)"
    exit 1
fi

echo "===== [2/2] venv内 decord 训练同路径冒烟 ====="
[ -f "${LOCAL}/venv/bin/activate" ] || { echo "FATAL: venv不存在, 先跑 setup_h20_venv.sh"; exit 1; }
source "${LOCAL}/venv/bin/activate"
python3 << 'PYEOF'
import os, sys
from multiprocessing import Pool
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

root = '/dockerdata/zhenjunchen/train_videos'
vids = []
for ds in ['cosmo_cap','didemo','internvid_vtime','queryd','hirest']:
    d = os.path.join(root, ds)
    if os.path.isdir(d):
        vids += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.mp4')]
vids.sort()
print(f"待测视频: {len(vids)} 个, 8进程并发, decord后端 fps=2采样中间帧")

def smoke(vp):
    try:
        from qwen_vl_utils import vision_process as vp_mod
        reader = vp_mod.VIDEO_READER_BACKENDS['decord']  # 训练同路径
        video, meta, sfps = reader({'video': vp, 'fps': 2.0, 'max_frames': 64})
        assert video.dim() == 4 and video.size(0) > 0, f"shape异常 {tuple(video.shape)}"
        return (vp, None)
    except Exception as e:
        return (vp, f"{type(e).__name__}: {e}")

with Pool(8) as p:
    results = p.map(smoke, vids)

fails = [(v, e) for v, e in results if e]
for v, e in fails[:8]:
    print(f"  FAIL: {v} -> {e}")
print(f"decord冒烟: {len(vids)-len(fails)}/{len(vids)} 可解码")
if fails:
    print("FATAL: decord在这台机器上不可靠 -> 训练必崩, 不要启动!")
    print("备选: 换机器, 或我帮你切PyAV/cv2后端")
    sys.exit(1)
print("decord冒烟全部通过 ✅")
PYEOF
if [ $? -ne 0 ]; then exit 1; fi

echo "============================================"
echo "验收完成: 视频完好 + decord训练路径可靠"
echo "可以启动训练: nohup bash ${CEPH}/code/Temporal-ZwZ/scripts/run_opd_tvg_h20_fixed_all_152.sh >> ${CEPH}/logs/run_fixed_all_152.log 2>&1 &"
echo "============================================"
