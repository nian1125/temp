#!/bin/bash
# ============================================================
# 噪声地板实验 seed=123 | teacher=numberit kl=1.0 (其余全同v8c)
# 目的: 与 seed=42(numberit=54.14) 组成3种子三元组, 测RL非确定性的std。
#       std就是"噪声地板", 之后任何改进必须 >~2σ 才算真。
# ============================================================
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
bash scripts/run_opd_tvg.sh \
    --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl" \
    --teacher_strategy numberit \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 200 \
    --save_steps 25 \
    --seed 123 \
    --name_suffix "-seed123"
