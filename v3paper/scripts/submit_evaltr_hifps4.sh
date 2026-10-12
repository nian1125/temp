#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# hifps4 在完整Charades-Bench评测(同54.99口径)
STEPS="200 150 250 100 300 350 50" bash scripts/eval_by_pattern.sh "opd-*-hifps4-sft-2026*" "hifps4-sft"
