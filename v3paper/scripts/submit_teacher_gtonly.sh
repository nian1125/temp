#!/bin/bash
# ============================================================
# Q8-10b 部分提示teacher | teacher=gt_only_numberit+red_border | kl=1.0
# ★高优先级: 只标注GT帧(编号+红框), 非GT帧无标注 —— 逼teacher自己补全推理。
#   按"最小特权+逼推理→最好student"的规律, 预期可能超过 numberit(54.14)。
# 对照: numberit 54.14 / 组合(n+rb) 54.47
# ============================================================
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

FILTERED_DATA="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl"

bash scripts/run_opd_tvg.sh \
    --anno_path "$FILTERED_DATA" \
    --teacher_strategy "gt_only_numberit+red_border" \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 200 \
    --save_steps 25
