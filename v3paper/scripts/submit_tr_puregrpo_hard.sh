#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 基线: hard难度采样数据 + 纯GRPO(无蒸馏)。对照 易数据纯GRPO 54.99 / TimeLens 55.2(同数据口径)。
bash scripts/run_opd_tvg.sh --model_path /primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/hard_2500.jsonl" --teacher_strategy numberit --kl_weight 0.0 --grpo_weight 1.0 --global_batch_size 8 --steps_per_generation 8 --loss_type grpo --num_generations 8 --max_steps 350 --save_steps 25 --reward_funcs tiou --name_suffix "-puregrpo-hard"
