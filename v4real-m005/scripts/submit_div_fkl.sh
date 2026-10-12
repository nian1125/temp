#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 散度对比: divergence=fkl (teacher=numberit, 其余同numberit基线)
# 对比: numberit JSD = 54.14。OPSD实测 fkl≫jsd≈rkl, 验证是否成立。
bash scripts/run_opd_tvg.sh \
    --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl" --teacher_strategy numberit \
    --kl_weight 1.0 --grpo_weight 1.0 --divergence fkl \
    --global_batch_size 8 --steps_per_generation 8 --loss_type grpo \
    --num_generations 8 --max_steps 200 --save_steps 25 \
    --name_suffix "-divfkl"
