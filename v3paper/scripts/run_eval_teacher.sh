#!/bin/bash
# Evaluate teacher strategies on TimeLens-Bench (multi-GPU)
#
# Usage:
#   bash scripts/run_eval_teacher.sh numberit+red_border
#   bash scripts/run_eval_teacher.sh gt_text
#   bash scripts/run_eval_teacher.sh gt_only_numberit+red_border
#   bash scripts/run_eval_teacher.sh all   # run all strategies sequentially
set -e

PROJECT_ROOT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3"
TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct"
BENCH_PATH="${TIMELENS_ROOT}/data/TimeLens-Bench/charades-timelens.json"
VIDEO_DIR="${TIMELENS_ROOT}/data/TimeLens-Bench/videos/charades"

export TIMELENS_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${TIMELENS_ROOT}:${PYTHONPATH:-}"
pip install decord pillow qwen-vl-utils imageio imageio-ffmpeg 2>/dev/null

STRATEGY="${1:-numberit+red_border}"

run_strategy() {
    local strategy="$1"
    echo ""
    echo "=========================================="
    echo "Evaluating strategy: ${strategy}"
    echo "=========================================="

    OUTPUT_DIR="${PROJECT_ROOT}/output/teacher_eval/${strategy}"
    mkdir -p "$OUTPUT_DIR"

    IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))}"
    CHUNKS=${#GPULIST[@]}
    echo "Using ${CHUNKS} GPUs: ${GPULIST[*]}"

    cd "$PROJECT_ROOT"

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python scripts/eval_teacher_strategies.py \
            --bench_path "$BENCH_PATH" \
            --video_dir "$VIDEO_DIR" \
            --model_path "$MODEL_PATH" \
            --teacher_strategy "$strategy" \
            --output_dir "$OUTPUT_DIR" \
            --chunk "$CHUNKS" \
            --index "$IDX" &
    done

    echo "Waiting for ${CHUNKS} processes..."
    wait
    echo "Inference done. Syncing filesystem..."
    sync
    sleep 5

    echo "Aggregating..."

    python scripts/eval_teacher_strategies.py --aggregate_dir "$OUTPUT_DIR"
}

if [ "$STRATEGY" = "all" ]; then
    for s in numberit+red_border gt_text gt_only_numberit+red_border; do
        run_strategy "$s"
    done
    echo ""
    echo "=========================================="
    echo "All strategies evaluated. Summary:"
    echo "=========================================="
    for s in numberit+red_border gt_text gt_only_numberit+red_border; do
        metrics="${PROJECT_ROOT}/output/teacher_eval/${s}/metrics.json"
        if [ -f "$metrics" ]; then
            echo "  ${s}: $(cat "$metrics")"
        fi
    done
else
    run_strategy "$STRATEGY"
fi
