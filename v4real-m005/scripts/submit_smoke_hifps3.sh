#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 均匀高fps teacher(zoom的公平实现, fps=3): teacher整段按3fps均匀采样(student默认2fps),
# 单video+正确fps元数据→Qwen时间编码不崩。JSD+GRPO, 同起点对照 sftinit_puregrpo 54.99。
export TEACHER_HIFPS=3
bash scripts/run_opd_tvg.sh --model_path /primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl" --teacher_strategy hifps --kl_weight 1.0 --grpo_weight 1.0 --global_batch_size 8 --steps_per_generation 8 --loss_type grpo --num_generations 8 --max_steps 15 --save_steps 100 --name_suffix "-hifps3-smoke"
