#!/usr/bin/env bash
# 在 H20 上执行: 把病态慢解码视频转码为 CFR 30fps + 密集关键帧, 治本解决 decord seek 极慢.
# 原视频备份到 ceph, 转码失败/损坏的写入 transcode_failures.txt 供剔除.
#
# 用法(在 H20):
#     bash scripts/fix_slow_videos.sh
#
# 依赖: ffmpeg / ffprobe
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUT="${PROJECT_ROOT}/output"

VIDEO_DIR="/dockerdata/zhenjunchen/train_videos"
TEACHER_CACHE="/dockerdata/zhenjunchen/teacher_cache"
BACKUP_DIR="${OUT}/train_videos_orig_backup"      # 原视频备份(ceph, 持久)
LIST="${OUT}/slow_source_videos.txt"              # 默认清单; 可用 $1 覆盖
FAIL="${OUT}/transcode_failures.txt"

# 可选: 第1个参数指定清单文件
[[ $# -ge 1 && -f "$1" ]] && LIST="$1"

command -v ffmpeg  >/dev/null 2>&1 || { echo "[FATAL] 未找到 ffmpeg"; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "[FATAL] 未找到 ffprobe"; exit 1; }
[[ -f "$LIST" ]] || { echo "[FATAL] 缺少清单 $LIST, 请先运行 scan_slow_videos.py"; exit 1; }

mkdir -p "$BACKUP_DIR"
: > "$FAIL"

# 用 mapfile 读取, 避免 while-read 管道问题
mapfile -t VIDEOS < "$LIST"
total=${#VIDEOS[@]}
i=0; ok=0; failn=0; missn=0
echo "共 ${total} 个待转码视频 (清单: $LIST)"

for bn in "${VIDEOS[@]}"; do
    bn="${bn%$'\r'}"   # 去掉可能的 \r
    bn="${bn// /}"      # 去掉空格
    [[ -z "$bn" ]] && continue
    i=$((i+1))
    src="${VIDEO_DIR}/${bn}"
    tmp="${src}.fixed.mp4"

    # 跳过已转过的(原视频已在backup里=已转成功)
    if [[ -f "${BACKUP_DIR}/${bn}" ]]; then
        echo "[$i/$total] SKIP     $bn  (已转码)"; continue
    fi

    if [[ ! -f "$src" ]]; then
        echo "[$i/$total] MISSING  $bn"; echo "$bn" >> "$FAIL"; missn=$((missn+1)); continue
    fi

    # CFR 30fps + 每15帧关键帧 + 无场景切换关键帧; 偶数分辨率; 去音轨; faststart
    ffmpeg -y -loglevel error -i "$src" \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        -r 30 -vsync cfr \
        -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
        -g 15 -keyint_min 15 -sc_threshold 0 \
        -an -movflags +faststart \
        "$tmp" 2>>"${OUT}/transcode_ffmpeg.err"

    # 校验产物有帧
    nf=$(ffprobe -v error -select_streams v:0 -count_packets \
         -show_entries stream=nb_read_packets -of csv=p=0 "$tmp" 2>/dev/null)
    if [[ -s "$tmp" && "${nf:-0}" =~ ^[0-9]+$ && "${nf:-0}" -gt 0 ]]; then
        mv -f "$src" "${BACKUP_DIR}/${bn}"     # 备份原视频
        mv -f "$tmp" "$src"                    # 用转码版替换
        ok=$((ok+1))
        echo "[$i/$total] OK       $bn  (frames=$nf)"
    else
        rm -f "$tmp"
        echo "[$i/$total] FAIL     $bn"; echo "$bn" >> "$FAIL"; failn=$((failn+1))
    fi
done

echo "============================================"
echo "转码成功: $ok  失败: $failn  缺失: $missn"
echo "失败/缺失清单(将被剔除): $FAIL"
echo "原视频备份目录:          $BACKUP_DIR"
echo ""
echo "重要: 清空 teacher_cache 以便按修复后的源视频重建:"
echo "    rm -f ${TEACHER_CACHE}/*.mp4"
echo "============================================"
