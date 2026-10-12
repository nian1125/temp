#!/bin/bash
# ============================================================
# 路线β 第2步: OPD GRPO+JSD 从 SFT checkpoint 热启动
# = 保留你的方法(JSD蒸馏), 只把起点从 base 换成 SFT checkpoint。
# teacher=numberit, 6838过滤数据, 配置全同 v8c。
# 对照: base起点 numberit = 54.14。看 SFT热启动是否涨点。
# ============================================================
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

SFT_CKPT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/sft/SFT-Qwen3-VL-8B-routeb"
DATA="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl"

if [ ! -d "$SFT_CKPT" ]; then
    echo "[ERROR] SFT checkpoint 不存在: $SFT_CKPT —— 请先跑 submit_sft_route_b.sh"
    exit 1
fi

bash scripts/run_opd_tvg.sh \
    --model_path "$SFT_CKPT" \
    --anno_path "$DATA" \
    --teacher_strategy numberit \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size 8 \
    --steps_per_generation 8 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps 200 \
    --save_steps 25 \
    --name_suffix "-fromSFT"
