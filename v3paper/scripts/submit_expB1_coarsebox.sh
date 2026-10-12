#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# B1(ZwZ核心机制时间版): student看粗时间框(GT×2窗口)+纯GRPO, 测试去掉(LUPI,标准eval)。
# 粗框只指方向不泄漏精确边界, student用tIoU精修。对比 纯GRPO=53.57
bash scripts/run_opd_tvg.sh --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl" --teacher_strategy numberit --kl_weight 0.0 --grpo_weight 1.0 --global_batch_size 8 --steps_per_generation 8 --loss_type grpo --num_generations 8 --max_steps 200 --save_steps 25 --student_mark_strategy coarse_box --name_suffix "-expB1-coarsebox"
