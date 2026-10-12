#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# Qwen2.5-VL 纯GRPO基线(无teacher), 先测它的天花板。注意base弱(plain 44.6 vs Qwen3 61.5)。
bash scripts/run_opd_tvg.sh \
    --model_path /primus_xpfs_root/zhangyajie/ckpt/Qwen2.5-VL-7B-Instruct \
    --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl" --teacher_strategy numberit \
    --kl_weight 0.0 --grpo_weight 1.0 \
    --global_batch_size 8 --steps_per_generation 8 --loss_type grpo \
    --num_generations 8 --max_steps 200 --save_steps 25 \
    --name_suffix "-qwen25base"
