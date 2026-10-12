#!/bin/bash
# =============================================================================
# OPD Marking Strategy Ablation Runner
# Runs all 6 marking strategies with OPD training
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VISION_OPD_ROOT="/cpfs01/user/chenzhaoyu.czy/code/Vision-OPD"

# --- Paths ---
DATA_DIR="/cpfs01/user/chenzhaoyu.czy/data/temporal_zwz"
VIDEO_DIR="/cpfs01/user/chenzhaoyu.czy/data/charades/videos_3fps_480_noaudio"
ANNO_TRAIN="/cpfs01/user/chenzhaoyu.czy/data/charades/charades_sta/charades_sta_train.txt"
ANNO_TEST="/cpfs01/user/chenzhaoyu.czy/data/charades/charades_sta/charades_sta_test.txt"
DURATIONS="/cpfs01/user/chenzhaoyu.czy/data/charades/charades_sta/durations.json"
RESULTS_DIR="${PROJECT_ROOT}/results"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"

# --- Model ---
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_SHORT="$(basename $MODEL_PATH)"

# --- Training hyperparams ---
NUM_TEACHER_FRAMES="${NUM_TEACHER_FRAMES:-16}"
PROMPT_STYLE="${PROMPT_STYLE:-grpo}"
NUM_WORKERS="${NUM_WORKERS:-16}"

# All 6 marking strategies
STRATEGIES=("numberit" "red_border" "t2sgrid" "timemarker" "numberit+red_border" "crop+numberit")

# Override with specific strategies if provided
if [ $# -gt 0 ]; then
    STRATEGIES=("$@")
fi

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR"

SUMMARY_FILE="${RESULTS_DIR}/ablation_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "=============================================" | tee "$SUMMARY_FILE"
echo "OPD Marking Strategy Ablation" | tee -a "$SUMMARY_FILE"
echo "Model: $MODEL_PATH" | tee -a "$SUMMARY_FILE"
echo "Teacher Frames: $NUM_TEACHER_FRAMES" | tee -a "$SUMMARY_FILE"
echo "Started: $(date)" | tee -a "$SUMMARY_FILE"
echo "=============================================" | tee -a "$SUMMARY_FILE"

export PYTHONPATH="$PROJECT_ROOT:$VISION_OPD_ROOT:${PYTHONPATH:-}"

for STRATEGY in "${STRATEGIES[@]}"; do
    SAFE_NAME="${STRATEGY//+/_}"
    echo ""
    echo ">>> Strategy: $STRATEGY" | tee -a "$SUMMARY_FILE"
    echo "    Started: $(date)" | tee -a "$SUMMARY_FILE"

    PARQUET="${DATA_DIR}/charades_train_${SAFE_NAME}.parquet"
    TEACHER_DIR="${DATA_DIR}/teacher_frames_${SAFE_NAME}"
    CKPT_DIR="${CHECKPOINT_DIR}/OPD-${MODEL_SHORT}-${SAFE_NAME}"
    PRED_FILE="${RESULTS_DIR}/preds_opd_${SAFE_NAME}.jsonl"

    # --- Step 1: Prepare data ---
    if [ ! -f "$PARQUET" ]; then
        echo "    [1/3] Preparing data..." | tee -a "$SUMMARY_FILE"
        python "${PROJECT_ROOT}/scripts/prepare_video_data.py" \
            --dataset charades \
            --anno_path "$ANNO_TRAIN" \
            --video_dir "$VIDEO_DIR" \
            --durations_path "$DURATIONS" \
            --output_path "$PARQUET" \
            --prompt_style "$PROMPT_STYLE" \
            --marking_strategy "$STRATEGY" \
            --preprocess_teacher \
            --teacher_output_dir "$TEACHER_DIR" \
            --num_teacher_frames "$NUM_TEACHER_FRAMES" \
            --num_workers "$NUM_WORKERS"
    else
        echo "    [1/3] Data exists, skipping." | tee -a "$SUMMARY_FILE"
    fi

    # --- Step 2: Train ---
    if [ ! -d "$CKPT_DIR" ] || [ ! -f "$CKPT_DIR/config.json" ]; then
        echo "    [2/3] Training OPD..." | tee -a "$SUMMARY_FILE"
        TRAIN_FILE="$PARQUET" \
        OUTPUT_DIR="$CKPT_DIR" \
        MODEL_PATH="$MODEL_PATH" \
        bash "${PROJECT_ROOT}/scripts/run_temporal_opd.sh"
    else
        echo "    [2/3] Checkpoint exists, skipping." | tee -a "$SUMMARY_FILE"
    fi

    # --- Step 3: Evaluate ---
    echo "    [3/3] Evaluating..." | tee -a "$SUMMARY_FILE"
    python "${PROJECT_ROOT}/scripts/eval_tvg.py" \
        --model_path "$CKPT_DIR" \
        --dataset charades \
        --anno_path "$ANNO_TEST" \
        --video_dir "$VIDEO_DIR" \
        --durations_path "$DURATIONS" \
        --output_file "$PRED_FILE" \
        --num_frames 32 --fps 2.0 \
        2>&1 | tee -a "$SUMMARY_FILE"

    echo "    Finished: $(date)" | tee -a "$SUMMARY_FILE"
    echo "---" | tee -a "$SUMMARY_FILE"
done

echo "" | tee -a "$SUMMARY_FILE"
echo "=============================================" | tee -a "$SUMMARY_FILE"
echo "All ablations complete: $(date)" | tee -a "$SUMMARY_FILE"
echo "Summary: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "=============================================" | tee -a "$SUMMARY_FILE"
