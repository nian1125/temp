#!/bin/bash
set -e
# 特权predictor on SFT模型 [2_coarsebox]: coarse_box:最低KL候选,峰值54.50。看SFT下KL是否仍最低
# 输出 Δ(有用度)+边界KL(可迁移性)。基座=官方SFT中间态。
export MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT"
export STRATEGIES="plain,coarse_box"
export TAG="sft-coarsebox"
export PATCH=16
export MAXS=60
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/submit_measure_gap.sh
