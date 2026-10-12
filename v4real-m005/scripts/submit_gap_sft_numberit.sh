#!/bin/bash
set -e
# 特权predictor on SFT模型 [1_numberit]: numberit:引擎teacher,学生最好,Δ曾=-1.14却最佳。看SFT下是否仍负
# 输出 Δ(有用度)+边界KL(可迁移性)。基座=官方SFT中间态。
export MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT"
export STRATEGIES="plain,numberit"
export TAG="sft-numberit"
export PATCH=16
export MAXS=60
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/submit_measure_gap.sh
