#!/bin/bash
set -e
# 特权predictor on SFT模型 [coarse辅助推理版]: coarse_text:gt_text非泄漏版(粗区域banner,不给精确答案)
export MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT"
export STRATEGIES="plain,coarse_text"
export TAG="sft-coarsetext"
export PATCH=16
export MAXS=60
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/submit_measure_gap.sh
