#!/bin/bash
# 评测 SFT 路线β 的 checkpoint (output/sft/, step 10-100)
# 看 SFT 后 base 的 TVG 定位能力, 挑最好的接 OPD 热启动
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
SFT_DIR="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/sft/SFT-Qwen3-VL-8B-routeb"
OUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"
cd "$TIMELENS_ROOT"; export PYTHONPATH="./:$PYTHONPATH"
export min_tokens=64 total_tokens=14336 FPS=2 datasets="charades-timelens"
RES="$OUT/eval_sft_results.txt"; echo "# SFT eval ($(date))" > "$RES"
for step in 100 90 80 70 60 50 40 30 20 10; do
    ckpt="$SFT_DIR/checkpoint-${step}"
    [ -d "$ckpt" ] || { echo "[SKIP] no $ckpt"; continue; }
    echo "==== SFT-step${step} ===="
    export model_path="$ckpt" pred_path="$OUT/eval_preds"
    bash scripts/eval_timelens_bench.sh 2>&1 | tee /tmp/eval_sft${step}.log
    echo "## SFT-step${step}" >> "$RES"
    grep -E "IOU 0\.[357]:|mIOU:" /tmp/eval_sft${step}.log | grep -v Warning | tail -4 >> "$RES"
done
echo "=== SFT eval done ==="; cat "$RES"
