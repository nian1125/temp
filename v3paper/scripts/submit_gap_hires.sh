#!/bin/bash
# 时间/空间超分 gap 测量 (Qwen3-VL)。配置写死, 直接 bash 本脚本即可。
P=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts
# 先拉最新 measure_gap.py + submit_measure_gap.sh
OSS="oss://quark-llm/primus/datasets/qianzi.qz/czy_transfer/scripts"
ossutil cp -f "$OSS/measure_gap.py" "$P/measure_gap.py" 2>/dev/null
ossutil cp -f "$OSS/submit_measure_gap.sh" "$P/submit_measure_gap.sh" 2>/dev/null
chmod +x "$P/submit_measure_gap.sh"
export STRATEGIES="plain,hires_frames,hires_pixels"
export MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct"
bash "$P/submit_measure_gap.sh"
