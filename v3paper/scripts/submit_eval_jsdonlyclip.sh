#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 崩溃实验早期checkpoint评测(看崩溃前能到多少)
bash scripts/eval_by_pattern.sh "opd-numberit+red_border-kl1.0-gw0.0-grpo-spg8-lr1e-6-jsdonly-clip1.0-*" "jsdonlyclip"
