#!/bin/bash
# Preprocess Charades videos: extract from zip → convert to 3fps, 480p, no audio
set -euo pipefail

ZIP_PATH="/cpfs01/user/chenzhaoyu.czy/data/charades/download/Charades_v1_480.zip"
EXTRACT_DIR="/cpfs01/user/chenzhaoyu.czy/data/charades/raw_480p"
OUTPUT_DIR="/cpfs01/user/chenzhaoyu.czy/data/charades/videos_3fps_480_noaudio"
NUM_WORKERS="${1:-8}"

mkdir -p "$EXTRACT_DIR" "$OUTPUT_DIR"

# Step 1: Extract
if [ ! -f "$EXTRACT_DIR/.done" ]; then
    echo "[1/2] Extracting $ZIP_PATH ..."
    7z x -o"$EXTRACT_DIR" "$ZIP_PATH" -y
    touch "$EXTRACT_DIR/.done"
    echo "  Extracted."
else
    echo "[1/2] Already extracted."
fi

# Find the actual video directory (might be nested)
VIDEO_SRC=$(find "$EXTRACT_DIR" -maxdepth 2 -name "*.mp4" -print -quit | xargs dirname)
echo "  Video source: $VIDEO_SRC"
echo "  Total videos: $(ls "$VIDEO_SRC"/*.mp4 2>/dev/null | wc -l)"

# Step 2: Convert to 3fps, 480p height, no audio
echo "[2/2] Converting to 3fps 480p no-audio ..."
convert_one() {
    local src="$1"
    local dst="$2/$(basename "$src")"
    [ -f "$dst" ] && return 0
    ffmpeg -y -i "$src" -an -vf "scale=-2:480,fps=3" -c:v libx264 -preset fast -crf 23 -loglevel error "$dst"
}
export -f convert_one

find "$VIDEO_SRC" -name "*.mp4" | \
    xargs -P "$NUM_WORKERS" -I{} bash -c "convert_one '{}' '$OUTPUT_DIR'"

DONE_COUNT=$(ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)
echo "  Done: $DONE_COUNT videos in $OUTPUT_DIR"
