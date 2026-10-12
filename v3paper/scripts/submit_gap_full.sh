#!/bin/bash
# 综合特权预测器: Qwen3-VL(主) + Qwen2.5-VL。配置写死, 直接 bash。
# 注意: 代码(measure_gap.py / tvg_opd_dataset.py)需先在登录节点 ossutil 拉好。
P=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts

echo "######## [1/2] Qwen3-VL-8B (主线) ########"
MODEL_PATH=/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct \
PATCH=16 MAXS=60 \
STRATEGIES=plain,numberit,timestamp,red_border,gt_text,coarse_box,hires_frames,hires_pixels,grid \
bash "$P/submit_measure_gap.sh" || echo "[WARN] Qwen3 failed"

echo "######## [2/2] Qwen2.5-VL-7B ########"
MODEL_PATH=/primus_xpfs_root/zhangyajie/ckpt/Qwen2.5-VL-7B-Instruct \
PATCH=14 MAXS=60 \
STRATEGIES=plain,numberit,timestamp,red_border,gt_text \
bash "$P/submit_measure_gap.sh" || echo "[WARN] Qwen2.5 failed"
echo "######## ALL GAP DONE ########"
