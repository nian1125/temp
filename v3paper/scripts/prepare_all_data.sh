#!/bin/bash
# Generate all training parquets (12 combinations: 6 strategies × 2 paradigms)
# Run AFTER videos are preprocessed to videos_3fps_480_noaudio/
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Paths ---
DATA_DIR="/cpfs01/user/chenzhaoyu.czy/data/temporal_zwz"
VIDEO_DIR="/cpfs01/user/chenzhaoyu.czy/data/charades/videos_3fps_480_noaudio"
ANNO_TRAIN="/cpfs01/user/chenzhaoyu.czy/data/charades/charades_sta/charades_sta_train.txt"
DURATIONS="/cpfs01/user/chenzhaoyu.czy/data/charades/charades_sta/durations.json"
NUM_FRAMES=32
NUM_TEACHER_FRAMES=16
NUM_WORKERS="${1:-16}"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

ALL_STRATEGIES=("numberit" "red_border" "t2sgrid" "timemarker" "numberit+red_border" "crop+numberit")

echo "============================================="
echo "Temporal-ZwZ: Generate Training Data"
echo "  Video dir: $VIDEO_DIR"
echo "  Strategies: ${#ALL_STRATEGIES[@]}"
echo "  Workers: $NUM_WORKERS"
echo "============================================="

# Check videos exist
VIDEO_COUNT=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "ERROR: No videos found in $VIDEO_DIR"
    echo "Run: bash scripts/preprocess_charades.sh"
    exit 1
fi
echo "Found $VIDEO_COUNT videos."

mkdir -p "$DATA_DIR"

# --- ZwZ-style: student sees MARKED video → RL ---
echo ""
echo "========== ZwZ-STYLE DATA (6 strategies) =========="
for strategy in "${ALL_STRATEGIES[@]}"; do
    safe="${strategy//+/_}"
    parquet="${DATA_DIR}/charades_zwz_${safe}.parquet"
    frames_dir="${DATA_DIR}/zwz_frames_${safe}"

    if [ -f "$parquet" ]; then
        echo "[SKIP] $parquet already exists"
        continue
    fi

    echo "[ZwZ] Generating: $strategy → $parquet"
    python "${PROJECT_ROOT}/scripts/prepare_zwz_data.py" \
        --anno_path "$ANNO_TRAIN" \
        --video_dir "$VIDEO_DIR" \
        --durations_path "$DURATIONS" \
        --output_path "$parquet" \
        --output_frames_dir "$frames_dir" \
        --marking_strategy "$strategy" \
        --num_frames "$NUM_FRAMES" \
        --num_workers "$NUM_WORKERS"
    echo "[ZwZ] Done: $parquet ($(wc -c < "$parquet" | numfmt --to=iec))"
done

# --- OPD-style: teacher sees MARKED clips → JSD → student sees clean ---
echo ""
echo "========== OPD-STYLE DATA (6 strategies) =========="
for strategy in "${ALL_STRATEGIES[@]}"; do
    safe="${strategy//+/_}"
    parquet="${DATA_DIR}/charades_opd_${safe}.parquet"
    teacher_dir="${DATA_DIR}/teacher_frames_${safe}"

    if [ -f "$parquet" ]; then
        echo "[SKIP] $parquet already exists"
        continue
    fi

    echo "[OPD] Generating: $strategy → $parquet"
    python "${PROJECT_ROOT}/scripts/prepare_video_data.py" \
        --dataset charades \
        --anno_path "$ANNO_TRAIN" \
        --video_dir "$VIDEO_DIR" \
        --durations_path "$DURATIONS" \
        --output_path "$parquet" \
        --prompt_style grpo \
        --marking_strategy "$strategy" \
        --preprocess_teacher \
        --teacher_output_dir "$teacher_dir" \
        --num_teacher_frames "$NUM_TEACHER_FRAMES" \
        --num_workers "$NUM_WORKERS"
    echo "[OPD] Done: $parquet ($(wc -c < "$parquet" | numfmt --to=iec))"
done

# --- Vanilla baseline (no marks) ---
echo ""
echo "========== BASELINE DATA =========="
VANILLA="${DATA_DIR}/charades_grpo_vanilla.parquet"
if [ ! -f "$VANILLA" ]; then
    echo "[Baseline] Generating vanilla GRPO data..."
    python "${PROJECT_ROOT}/scripts/prepare_video_data.py" \
        --dataset charades \
        --anno_path "$ANNO_TRAIN" \
        --video_dir "$VIDEO_DIR" \
        --durations_path "$DURATIONS" \
        --output_path "$VANILLA" \
        --prompt_style grpo \
        --marking_strategy numberit \
        --num_teacher_frames "$NUM_TEACHER_FRAMES" \
        --num_workers "$NUM_WORKERS"
fi

echo ""
echo "========== SUMMARY =========="
echo "Generated parquets:"
ls -lh "${DATA_DIR}"/charades_*.parquet 2>/dev/null
echo ""
echo "Total: $(ls "${DATA_DIR}"/charades_*.parquet 2>/dev/null | wc -l) parquet files"
echo "Next: bash scripts/run_all_experiments.sh all"
