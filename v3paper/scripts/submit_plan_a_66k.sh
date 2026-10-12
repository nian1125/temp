#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# Plan A 全量: 66K 去毒数据(vs 当前6.8K=54.47)。numberit+kl1, 400步(数据多更耐训)。
bash scripts/run_opd_tvg.sh \
    --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores/filtered_teacher_gt0.3.jsonl" --teacher_strategy numberit \
    --kl_weight 1.0 --grpo_weight 1.0 --global_batch_size 8 --steps_per_generation 8 \
    --loss_type grpo --num_generations 8 --max_steps 400 --save_steps 50 \
    --name_suffix "-planA66k"
