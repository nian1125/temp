#!/bin/bash
# 一条命令顺序跑完所有 gap 测量 (Qwen3新候选 + Qwen2.5全套)
P=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts
echo "######## [1/2] Qwen3-VL: plain,grid,coarse_box ########"
STRATEGIES=plain,grid,coarse_box bash $P/submit_measure_gap.sh || echo "[WARN] Qwen3 run failed"
echo "######## [2/2] Qwen2.5-VL: plain,numberit,timestamp,red_border,gt_text,grid ########"
MODEL_PATH=/primus_xpfs_root/zhangyajie/ckpt/Qwen2.5-VL-7B-Instruct \
STRATEGIES=plain,numberit,timestamp,red_border,gt_text,grid \
bash $P/submit_measure_gap.sh || echo "[WARN] Qwen2.5 run failed"
echo "######## ALL GAP DONE ########"
