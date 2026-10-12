#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 纯OPD-redborder(Line1, 对标Video-OPD)在完整Charades评测
STEPS="200 150 250 100 300 350 50" bash scripts/eval_by_pattern.sh "opd-*-pureopd-redborder-2026*" "pureopd-redborder"
