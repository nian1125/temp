#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
export PYTHONPATH="./:${PYTHONPATH:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 在【你的harness】测官方TimeLens-8B的ANet/QVH(你之前只测了它的Charades=55.34)
# → 拿到真正apples-to-apples的对比基准
export model_path=/primus_xpfs_root/zhangyajie/ckpt/TimeLens-8B
export datasets="activitynet-timelens,qvhighlights-timelens"
export min_tokens=64 total_tokens=14336 FPS=2
export pred_path=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/eval-timelens8b-anetqvh
bash scripts/eval_timelens_bench.sh
