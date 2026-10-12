#!/bin/bash
set -e
# 特权predictor on SFT模型 [4_redborder]: red_border:精确GT红框(强泄漏),对照coarse_box
# 输出 Δ(有用度)+边界KL(可迁移性)。基座=官方SFT中间态。
export MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT"
export STRATEGIES="plain,red_border"
export TAG="sft-redborder"
export PATCH=16
export MAXS=60
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/submit_measure_gap.sh
