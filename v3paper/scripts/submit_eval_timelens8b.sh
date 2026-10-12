#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
export PYTHONPATH="./:${PYTHONPATH:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 验官方TimeLens-8B在Charades是否=55.2 → 验证我的eval harness。复现到55.2则54.99 vs 55.2对比坐实; 复现不出则harness有偏差需先修。
export model_path=/primus_xpfs_root/zhangyajie/ckpt/TimeLens-8B
export datasets=charades-timelens
export min_tokens=64
export total_tokens=14336
export FPS=2
export pred_path=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/eval-timelens8b-official
bash scripts/eval_timelens_bench.sh
