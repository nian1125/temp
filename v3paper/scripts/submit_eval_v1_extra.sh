#!/bin/bash
cd /primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
pip install decord pillow qwen-vl-utils nncore 2>/dev/null
bash scripts/eval_v1_extra.sh
