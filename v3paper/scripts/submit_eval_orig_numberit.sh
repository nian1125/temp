#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 重评原始numberit(0625-1345, 当初报54.14的那个模型), 用当前评测pipeline。
# 分离"训练问题vs评测问题": 重评还54=训练问题; 变52=评测artifact。
bash scripts/eval_by_pattern.sh "opd-numberit-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260625-1345" "orig-numberit-recheck"
