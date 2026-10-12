#!/bin/bash
# ============================================================
# Q3 kl扫描 | kl_weight=0.0 (纯GRPO, 无JSD) | teacher=numberit
# 目的: 直接回答"JSD是否有用" —— kl=0 即关掉蒸馏。
#       若接近 numberit baseline(54.14) 说明JSD贡献小; 若明显掉点说明JSD关键。
# 对照: numberit kl=1.0 = 54.14
# ============================================================
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

FILTERED_DATA="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl"

bash scripts/run_opd_tvg.sh \
    --anno_path "$FILTERED_DATA" \
    --teacher_strategy numberit \
    --kl_weight 0.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 200 \
    --save_steps 25 \
    --name_suffix "-klsweep"
