#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
pip install bitsandbytes 2>/dev/null   # ExOPD 4-bit ref 需要
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# ExOPD λ外推: divergence=exopd, λ=1.25。
# 用独立 4-bit ref(~4GB/卡, 不经deepspeed) → 默认 zero2 即可, 不再OOM/不用zero3。
# beta=0 (不走父类ref)。teacher=numberit。对比 numberit JSD=54.14。
MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct"
bash scripts/run_opd_tvg.sh \
    --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl" --teacher_strategy numberit \
    --kl_weight 1.0 --grpo_weight 1.0 --divergence exopd \
    --exopd_lambda 1.25 --beta 0.0 \
    --exopd_ref_path "$MODEL_PATH" \
    --global_batch_size 8 --steps_per_generation 8 --loss_type grpo \
    --num_generations 8 --max_steps 200 --save_steps 25 \
    --name_suffix "-exopd1.25"
