#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
export PYTHONPATH="./:${PYTHONPATH:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 前置(忠实复现必需): 给全量 gemini_refined_data(96K)打base-iou。几小时GPU。
bash scripts/filter_data/filter_data_qwen3_vl.sh --dataset gemini_refined_data --model_path /primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct
mkdir -p /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/tl_filter
SRC=$(ls -t output/TimeLens-8B/filter-data/*/gemini_refined_data.jsonl 2>/dev/null | head -1)
cp "$SRC" "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/tl_filter/gemini_iou_full.jsonl"
echo "[OK] 全量iou数据 -> /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/tl_filter/gemini_iou_full.jsonl"
wc -l "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/tl_filter/gemini_iou_full.jsonl"
