#!/bin/bash
# ============================================================
# 路线β 第1步 提交: SFT 热启动 (Qwen3-VL-8B)
# ⚠️ 先冒烟! 见 run_sft_route_b.sh 顶部提醒(train_sft.py原为Qwen2.5-VL写)
# ============================================================
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
bash scripts/run_sft_route_b.sh
