#!/bin/bash
# ============================================================
# kl_weight schedule 实验: 前低后高 (kl 0.5 → 2.0 线性)
# 动机(来自v8c日志启示): 后期GRPO信号枯竭(zero_std 1.6%→36%, reward_std腰斩),
#   但JSD后期仍有效(低JSD对应高reward)。故前期给GRPO探索空间(kl低),
#   后期GRPO没料时加大蒸馏接力(kl高)。
# teacher=numberit, 6838, 其余同v8c。对照: 固定kl=1.0 numberit=54.14。
# ============================================================
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

DATA="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl"

bash scripts/run_opd_tvg.sh \
    --anno_path "$DATA" \
    --teacher_strategy numberit \
    --kl_weight 0.5 \
    --kl_weight_end 2.0 \
    --kl_schedule linear \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 200 \
    --save_steps 25 \
    --name_suffix "-klsched0.5to2.0"
