#!/bin/bash

cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3

# --- Environment setup ---
pip install -r requirements.txt
pip show decord > /dev/null 2>&1 || pip install decord
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# Flash-attention
pip show flash-attn > /dev/null 2>&1
if [ $? -ne 0 ]; then
    WHEEL="/primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    if [ -f "$WHEEL" ]; then
        pip install "$WHEEL"
    else
        pip install flash-attn --no-build-isolation
    fi
fi

# --- Paths ---
MODEL=/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct
TIMELENS_ROOT=/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens/data/TimeLens-Bench
ANNO=${TIMELENS_ROOT}/charades-timelens.json
VIDEO_DIR=${TIMELENS_ROOT}/videos/charades

# v3 fix: only re-run crop+numberit (other 3 strategies already have valid results in v2)
STRATEGIES="crop+numberit"

# 200 samples for quick validation; 0 = full (~3000+ query-span pairs)
SAMPLES=200

echo "=========================================="
echo "Zero-Shot TVG: Fix crop+numberit (v3)"
echo "Model:      Qwen3-VL-8B-Instruct"
echo "Dataset:    TimeLens-Bench Charades"
echo "Strategies: ${STRATEGIES}"
echo "Samples:    ${SAMPLES}"
echo "=========================================="

python scripts/zero_shot_tvg_test.py \
    --model_path $MODEL \
    --dataset timelens_charades \
    --anno_path $ANNO \
    --video_dir $VIDEO_DIR \
    --strategies $STRATEGIES \
    --max_samples $SAMPLES \
    --output_dir results/zero_shot_tvg_qwen3_v2/ \
    --fps 2 \
    --no_think \
    --force

echo "=========================================="
echo "Done. Results: results/zero_shot_tvg_qwen3/"
echo "=========================================="
