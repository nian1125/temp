#!/bin/bash
# OPD v4: 真 GRPO (steps_per_generation=4, loss_type=grpo)
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils imageio imageio-ffmpeg opencv-python-headless 2>/dev/null
bash scripts/run_opd_tvg_v4.sh
