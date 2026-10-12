#!/bin/bash
set -e
# Charades-Bench 特权predictor(并行单策略) [red_border], SFT基座
export MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT"
export ANNO="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/charades_bench_gap.jsonl"
export VIDEO_DIR="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
export STRATEGIES="plain,red_border"
export TAG="benchg-redborder"
export PATCH=16
export MAXS=50
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/submit_measure_gap.sh
