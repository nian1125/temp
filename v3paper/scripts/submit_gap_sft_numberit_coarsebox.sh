#!/bin/bash
set -e
# 特权predictor on SFT模型 [coarse辅助推理版]: numberit+coarse_box:v8c非泄漏版(帧号+粗窗口红框)
export MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT"
export STRATEGIES="plain,numberit+coarse_box"
export TAG="sft-nb_coarsebox"
export PATCH=16
export MAXS=60
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/submit_measure_gap.sh
