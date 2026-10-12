#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
export PYTHONPATH="./:$PYTHONPATH"
export model_path=/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct
export datasets="charades-timelens"
export pred_path=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/evalbench_hires
export min_tokens=64
export total_tokens=28672
export FPS=2
# base Qwen3 zero-shot on charades测试集, hires(FPS=2, total_tokens=28672)
bash scripts/eval_timelens_bench.sh
