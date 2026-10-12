#!/bin/bash
set -e
# 特权predictor on SFT模型 [7_numberit_redborder]: numberit+red_border:v8c组合(曾54.47)
# 输出 Δ(有用度)+边界KL(可迁移性)。基座=官方SFT中间态。
export MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT"
export STRATEGIES="plain,numberit+red_border"
export TAG="sft-nb_redborder"
export PATCH=16
export MAXS=60
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/submit_measure_gap.sh
