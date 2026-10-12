#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# timestamp teacher: 每帧叠真实秒数(GT-free)。对比 numberit(帧编号)=53.83
# 验证"工具形态": 直接给秒数 vs 给帧编号(需index→time换算), 哪个更可迁移
bash scripts/run_opd_tvg.sh \
    --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl" --teacher_strategy timestamp \
    --kl_weight 1.0 --grpo_weight 1.0 \
    --global_batch_size 8 --steps_per_generation 8 --loss_type grpo \
    --num_generations 8 --max_steps 200 --save_steps 25
