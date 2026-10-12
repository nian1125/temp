#!/bin/bash
# 评测 JSD-only (grpo_weight=0, teacher=numberit+red_border)
# 训练已崩溃(reward后期→0,jsd爆炸)。评全部8个checkpoint, 重点看早期(25/50/75 崩溃前)能到多少。
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
bash scripts/eval_by_pattern.sh "opd-numberit+red_border-kl1.0-gw0.0-*" "jsdonly"
