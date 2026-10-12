#!/usr/bin/env python3
"""诊断: charades测试集里, 多少视频"帧够"支持高fps采样。
判据: 在 fps=F 下, 请求帧数 = 时长×F; 若 实际帧数 < 请求帧数 → 帧不够(高fps空操作)。
本机直接跑(decord, 无需GPU)。
"""
import json, os, sys
import decord
import numpy as np

BENCH = "/cpfs01/user/chenzhaoyu.czy/data/timelens_bench/charades-timelens.json"
VIDDIR = "/cpfs01/user/chenzhaoyu.czy/data/charades/raw_480p/Charades_v1_480"
FPS_TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
FPS_BASE = 2.0
MAXN = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # 0=全部

data = json.load(open(BENCH))
vids = list(data.keys())
if MAXN: vids = vids[:MAXN]

native_fps = []
enough_hi = 0     # fps=4 下帧够
enough_base = 0   # fps=2 下帧够
limited_even_base = 0  # 连fps=2都不够
missing = 0
n = 0
ratios = []   # 实际帧数 / (时长×4)
for vid in vids:
    p = os.path.join(VIDDIR, f"{vid}.mp4")
    if not os.path.exists(p):
        missing += 1; continue
    try:
        vr = decord.VideoReader(p)
        tf = len(vr)
        fps = vr.get_avg_fps()
        dur = tf / fps if fps > 0 else 0
    except Exception:
        missing += 1; continue
    if dur <= 0: continue
    n += 1
    native_fps.append(fps)
    req_hi = dur * FPS_TARGET
    req_base = dur * FPS_BASE
    ratios.append(tf / req_hi if req_hi > 0 else 0)
    if tf >= req_hi: enough_hi += 1
    if tf >= req_base: enough_base += 1
    else: limited_even_base += 1

print(f"=== charades测试集 高fps可行性诊断 (目标fps={FPS_TARGET}) ===")
print(f"总视频: {len(vids)}  可读: {n}  缺失/坏: {missing}")
if n:
    nf = np.array(native_fps)
    print(f"\n源视频原生fps: mean={nf.mean():.1f} median={np.median(nf):.1f} min={nf.min():.1f} max={nf.max():.1f}")
    print(f"  原生fps<4 的视频: {(nf<4).sum()}/{n} ({(nf<4).mean()*100:.1f}%)  ← 这些根本采不到fps4")
    print(f"\nfps={FPS_TARGET} 帧够(实际帧≥时长×{FPS_TARGET}): {enough_hi}/{n} ({enough_hi/n*100:.1f}%)  ← 高fps真能生效的样本")
    print(f"fps={FPS_BASE} 帧够: {enough_base}/{n} ({enough_base/n*100:.1f}%)")
    print(f"连fps={FPS_BASE}都不够: {limited_even_base}/{n} ({limited_even_base/n*100:.1f}%)")
    r = np.array(ratios)
    print(f"\n实际帧/(时长×{FPS_TARGET}) 分布: median={np.median(r):.2f} (≥1=帧够)")
    print(f"\n>>> 结论: 高fps实验在 {enough_hi/n*100:.0f}% 的样本上能生效。" +
          ("  占比高→值得做(在帧够子集上看Δ)。" if enough_hi/n>0.5 else "  占比低→数据受限, 高fps基本无意义, 建议跳过。"))
