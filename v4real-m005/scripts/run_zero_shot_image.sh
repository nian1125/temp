#!/bin/bash

cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ

# --- Environment setup ---
pip install -r requirements.txt
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python --version
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# Flash-attention (use pre-built wheel if available)
pip show flash-attn > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing flash-attn..."
    WHEEL="/primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    if [ -f "$WHEEL" ]; then
        pip install "$WHEEL"
    else
        pip install flash-attn --no-build-isolation
    fi
fi
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"

# --- Download data via mirror (if not already present) ---
if [ ! -d "data/benchmarks/hrbench-4k" ]; then
    echo "Downloading benchmark data via HF mirror..."
    export HF_ENDPOINT=https://hf-mirror.com
    python -c "
from datasets import load_dataset
from huggingface_hub import snapshot_download
import os
os.makedirs('data/benchmarks', exist_ok=True)
print('HR-Bench 4K...')
load_dataset('DreamMr/HR-Bench', 'hrbench_version_split', split='hrbench_4k').save_to_disk('data/benchmarks/hrbench-4k')
print('HR-Bench 8K...')
load_dataset('DreamMr/HR-Bench', 'hrbench_version_split', split='hrbench_8k').save_to_disk('data/benchmarks/hrbench-8k')
print('V*Bench...')
load_dataset('craigwu/vstar_bench', split='test').save_to_disk('data/benchmarks/vstar')
snapshot_download('craigwu/vstar_bench', repo_type='dataset', local_dir='data/benchmarks/vstar_images')
print('Done!')
"
fi

# --- Run zero-shot image test ---
MODEL=/primus_xpfs_root/zhangyajie/ckpt/Qwen2.5-VL-7B-Instruct

STRATEGIES="original,A_numberit,B_coordinate,D_number_coord,E_sliding_grid"

# --- Test 1: hrbench-4k without thumbnail ---
echo "=========================================="
echo "[1/4] hrbench-4k (no thumbnail)"
echo "=========================================="
python scripts/zero_shot_image_test.py \
    --model_path $MODEL \
    --benchmark hrbench-4k \
    --data_dir data/benchmarks \
    --strategies $STRATEGIES \
    --grid_rows 3 --grid_cols 3 --overlap 0.15 \
    --output_dir results/zero_shot_image_4k/

# --- Test 2: hrbench-8k without thumbnail ---
echo "=========================================="
echo "[2/4] hrbench-8k (no thumbnail)"
echo "=========================================="
python scripts/zero_shot_image_test.py \
    --model_path $MODEL \
    --benchmark hrbench-8k \
    --data_dir data/benchmarks \
    --strategies $STRATEGIES \
    --grid_rows 3 --grid_cols 3 --overlap 0.15 \
    --output_dir results/zero_shot_image_8k/

# --- Test 3: hrbench-4k WITH thumbnail ---
echo "=========================================="
echo "[3/4] hrbench-4k + global thumbnail"
echo "=========================================="
python scripts/zero_shot_image_test.py \
    --model_path $MODEL \
    --benchmark hrbench-4k \
    --data_dir data/benchmarks \
    --strategies $STRATEGIES \
    --grid_rows 3 --grid_cols 3 --overlap 0.15 \
    --output_dir results/zero_shot_image_4k_thumb/ \
    --add_thumbnail --thumbnail_size 768

# --- Test 4: hrbench-8k WITH thumbnail ---
echo "=========================================="
echo "[4/4] hrbench-8k + global thumbnail"
echo "=========================================="
python scripts/zero_shot_image_test.py \
    --model_path $MODEL \
    --benchmark hrbench-8k \
    --data_dir data/benchmarks \
    --strategies $STRATEGIES \
    --grid_rows 3 --grid_cols 3 --overlap 0.15 \
    --output_dir results/zero_shot_image_8k_thumb/ \
    --add_thumbnail --thumbnail_size 768

echo "=========================================="
echo "All Qwen2.5-VL tests complete."
echo "Results:"
echo "  4k:          results/zero_shot_image_4k/"
echo "  8k:          results/zero_shot_image_8k/"
echo "  4k+thumb:    results/zero_shot_image_4k_thumb/"
echo "  8k+thumb:    results/zero_shot_image_8k_thumb/"
echo "=========================================="
