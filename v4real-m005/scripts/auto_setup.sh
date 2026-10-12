#!/bin/bash
# Automated pipeline: wait for download → extract → symlink → generate parquets
set -euo pipefail

ZIP="/cpfs01/user/chenzhaoyu.czy/data/charades/download/Charades_v1_480.zip"
EXTRACT="/cpfs01/user/chenzhaoyu.czy/data/charades/raw_480p"
VIDEO_LINK="/cpfs01/user/chenzhaoyu.czy/data/charades/videos_3fps_480_noaudio"
EXPECTED_SIZE=16339546533

echo "[1/4] Waiting for download to complete..."
while true; do
    [ -f "$ZIP" ] || { sleep 5; continue; }
    current=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
    if [ "$current" -ge "$EXPECTED_SIZE" ]; then
        echo "  Download complete: $(du -h "$ZIP" | cut -f1)"
        break
    fi
    pct=$((current * 100 / EXPECTED_SIZE))
    echo "  ${pct}%..."
    sleep 30
done

# Wait a moment for the file to be fully closed
sleep 5

echo "[2/4] Extracting with 7z..."
rm -rf "$EXTRACT"
mkdir -p "$EXTRACT"
7z x -o"$EXTRACT" "$ZIP" -y
VIDEO_DIR="$EXTRACT/Charades_v1_480"
count=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l)
echo "  Extracted: $count videos"

echo "[3/4] Creating symlink..."
rm -rf "$VIDEO_LINK"
ln -sfn "$VIDEO_DIR" "$VIDEO_LINK"
echo "  $VIDEO_LINK → $VIDEO_DIR"

echo "[4/4] Verifying coverage..."
python -c "
import os
video_dir = '$VIDEO_DIR'
available = set(f.replace('.mp4','') for f in os.listdir(video_dir) if f.endswith('.mp4'))
needed = set()
for split in ['train', 'test']:
    with open(f'/cpfs01/user/chenzhaoyu.czy/data/charades/charades_sta/charades_sta_{split}.txt') as f:
        for line in f:
            needed.add(line.strip().split()[0])
have = needed & available
print(f'Available: {len(available)} | Needed: {len(needed)} | Have: {len(have)} ({len(have)/len(needed)*100:.0f}%)')
missing = len(needed) - len(have)
if missing > 0:
    print(f'WARNING: {missing} videos missing')
"

echo ""
echo "============================="
echo "Done! Next steps:"
echo "  1. Generate training data: bash scripts/prepare_all_data.sh 16"
echo "  2. Run experiments: bash scripts/run_all_experiments.sh all"
echo "============================="
