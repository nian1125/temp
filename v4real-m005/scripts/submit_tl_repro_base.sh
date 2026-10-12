#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
export PYTHONPATH="./:${PYTHONPATH:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# Exp D: TimeLens配方从base Instruct(无SFT) — 隔离SFT-revert价值。需先跑 filter_full。
bash train_scripts/run_grpo_and_eval_qwen3_8b.sh --model_path /primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct --raw_anno_path /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/tl_filter/gemini_iou_full.jsonl --target_size 2500 --global_batch_size 64 --job_name repro-base
