#!/bin/bash
# OPD v5: Teacher Quality Gate (线性 gate)
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils imageio imageio-ffmpeg opencv-python-headless 2>/dev/null
bash scripts/run_opd_tvg.sh \
    --teacher_quality_gate True \
    --kl_weight 1.0 \
    --max_steps 300 \
    --save_steps 50
