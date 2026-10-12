#!/bin/bash
# Download ActivityNet videos needed for TimeLens-Bench evaluation
# Videos are from YouTube, downloaded via yt-dlp
set -uo pipefail

OUTPUT_DIR="/cpfs01/user/chenzhaoyu.czy/data/anet/videos"
ANNO_FILE="/cpfs01/user/chenzhaoyu.czy/data/timelens_bench/activitynet-timelens.json"
NUM_WORKERS="${1:-4}"
LOG_FILE="/cpfs01/user/chenzhaoyu.czy/data/anet/download_log.txt"

mkdir -p "$OUTPUT_DIR"

# Extract video IDs (format: v_YOUTUBEID → YOUTUBEID)
python -c "
import json
with open('$ANNO_FILE') as f:
    data = json.load(f)
for vid in sorted(data.keys()):
    yt_id = vid.replace('v_', '')
    print(yt_id)
" > /tmp/anet_video_ids.txt

TOTAL=$(wc -l < /tmp/anet_video_ids.txt)
echo "Total videos to download: $TOTAL" | tee "$LOG_FILE"

download_one() {
    local yt_id="$1"
    local output_dir="$2"
    local outfile="${output_dir}/v_${yt_id}.mp4"

    [ -f "$outfile" ] && return 0

    yt-dlp -f "best[height<=480][ext=mp4]/best[height<=480]/best[ext=mp4]/best" \
        --no-playlist --no-warnings --quiet \
        -o "$outfile" \
        "https://www.youtube.com/watch?v=${yt_id}" 2>/dev/null && \
        echo "OK: v_${yt_id}" || echo "FAIL: v_${yt_id}"
}
export -f download_one

cat /tmp/anet_video_ids.txt | \
    xargs -P "$NUM_WORKERS" -I{} bash -c "download_one '{}' '$OUTPUT_DIR'" 2>&1 | tee -a "$LOG_FILE"

DONE=$(ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)
echo "Downloaded: $DONE / $TOTAL" | tee -a "$LOG_FILE"
