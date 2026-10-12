#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
export PYTHONPATH="./:${PYTHONPATH:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# SFT裸baseline: TimeLens官方SFT中间态(JungleGym下载)在Charades裸测(不训练)。对照: TimeLens-8B最终55.2 / sftinit+纯GRPO峰值54.99。预期SFT本身约50-53(中间态应低于最终)。
export model_path=/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT
export datasets=charades-timelens
export min_tokens=64
export total_tokens=14336
export FPS=2
export pred_path=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/eval-sft-baseline
bash scripts/eval_timelens_bench.sh
