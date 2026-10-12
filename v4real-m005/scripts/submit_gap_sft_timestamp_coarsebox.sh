#!/bin/bash
set -e
# 特权predictor on SFT模型 [coarse辅助推理版]: timestamp+coarse_box:时间戳+粗窗口红框(不给精确答案)
export MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT"
export STRATEGIES="plain,timestamp+coarse_box"
export TAG="sft-ts_coarsebox"
export PATCH=16
export MAXS=60
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/submit_measure_gap.sh
