#!/bin/bash
# ============================================================
# Q8-13 advantage std归一化 | scale_rewards=True | teacher=numberit | kl=1.0
# 目的: 当前默认 scale_rewards=False(不除std, Vision-OPD设置)。本实验翻转成True,
#       看 std 归一化会不会改变效果。
# 对照: numberit kl=1.0 scale_rewards=False = 54.14
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
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 200 \
    --save_steps 25 \
    --scale_rewards True \
    --name_suffix "-scalerwTrue"
