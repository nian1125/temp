#!/bin/bash
# v8a-short + v8b-short 评估提交脚本
# 提交到集群运行，评测 16 个 checkpoint (v8a-short 8 + v8b-short 8)
# 预计耗时: ~4-5 小时 (每个 checkpoint ~15-20 分钟)
set -e

cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3

# 安装依赖
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# 运行评估
bash scripts/eval_v8_short.sh
