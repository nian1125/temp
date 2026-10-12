#!/bin/bash
set -e
# 特权predictor on SFT模型 [3_coarseboxhint]: coarse_box_hint:红框+温和提示变体
# 输出 Δ(有用度)+边界KL(可迁移性)。基座=官方SFT中间态。
export MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT"
export STRATEGIES="plain,coarse_box_hint"
export TAG="sft-coarseboxhint"
export PATCH=16
export MAXS=60
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/submit_measure_gap.sh
