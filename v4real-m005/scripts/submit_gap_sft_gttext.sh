#!/bin/bash
set -e
# 特权predictor on SFT模型 [5_gttext]: gt_text:答案直给(最强泄漏),Δ上界参照
# 输出 Δ(有用度)+边界KL(可迁移性)。基座=官方SFT中间态。
export MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT"
export STRATEGIES="plain,gt_text"
export TAG="sft-gttext"
export PATCH=16
export MAXS=60
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/submit_measure_gap.sh
