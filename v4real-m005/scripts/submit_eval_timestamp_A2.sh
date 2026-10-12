#!/bin/bash
# A2: gt_text+timestamp 策略评测
set -e

cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3

# 安装依赖
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# gt_text+timestamp 策略评测
bash scripts/run_eval_teacher.sh gt_text+timestamp
