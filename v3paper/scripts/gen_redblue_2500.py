#!/usr/bin/env python3
"""为 JSONL 数据逐样本生成红蓝框视频。
   key 格式与 eval_dataloader.py 一致: rb_{md5}.mp4
   用法: python scripts/gen_redblue_2500.py --anno <path> --cache <dir>
"""
import argparse, hashlib, json, os
from collections import defaultdict

import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw

BORDER_W = 8


def _add_border(frame, color, width=BORDER_W):
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    w, h = frame.size
    for i in range(width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
    return frame


def _read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", required=True, help="JSONL annotation file")
    ap.add_argument("--cache", required=True, help="Cache directory for redblue videos")
    ap.add_argument("--video_dir", default=None, help="Base dir for relative video paths")
    args = ap.parse_args()
    ANNO_PATH = args.anno
    CACHE_DIR = args.cache
    VIDEO_DIR = args.video_dir

    with open(ANNO_PATH) as f:
        annos = [json.loads(line) for line in f if line.strip()]

    video_groups = defaultdict(list)
    for a in annos:
        video_groups[a["video_path"]].append(a)

    # 收集每个视频的所有 span（用于 key 和蓝框）
    video_spans = defaultdict(list)
    for a in annos:
        sp = a["span"]
        gt = sp[0] if isinstance(sp[0], (list, tuple)) else sp
        video_spans[a["video_path"]].append((float(gt[0]), float(gt[1])))

    print(f"Samples: {len(annos)}, Unique videos: {len(video_groups)}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 缓存 key 格式（与 eval_dataloader.py 一致）
    def _cache_path(anno, video_spans):
        vp = anno["video_path"]
        sp = anno["span"]
        gt = sp[0] if isinstance(sp[0], (list, tuple)) else sp
        gt = [float(gt[0]), float(gt[1])]
        all_span_list = sorted(video_spans[vp])
        other_key = "-".join([f"{s[0]:.1f}_{s[1]:.1f}" for s in all_span_list])
        key = hashlib.md5(
            f"{vp}_{gt[0]:.2f}_{gt[1]:.2f}_1.5_{other_key}".encode()
        ).hexdigest()[:16]
        return os.path.join(CACHE_DIR, f"rb_{key}.mp4")

    # 辅助：解析相对路径（video_shards/{ds}/{ds}/{video}.mp4 多一层嵌套）
    def _resolve(vp):
        if os.path.isabs(vp):
            return vp
        if VIDEO_DIR:
            parts = vp.split("/", 1)
            if len(parts) == 2:
                return os.path.join(VIDEO_DIR, parts[0], parts[0], parts[1])
            return os.path.join(VIDEO_DIR, vp)
        return vp

    done = 0
    for video_path, group in video_groups.items():
        real_vp = _resolve(video_path)
        try:
            frames, fps = _read_frames(real_vp)
        except Exception as e:
            print(f"  SKIP {video_path}: {e}")
            continue

        N = len(frames)
        all_spans_map = {}
        for a in group:
            sp = a["span"]
            gt = sp[0] if isinstance(sp[0], (list, tuple)) else sp
            all_spans_map[a["query"]] = [float(gt[0]), float(gt[1])]

        for a in group:
            out_path = _cache_path(a, video_spans)
            if os.path.exists(out_path):
                done += 1
                continue

            sp = a["span"]
            gt = sp[0] if isinstance(sp[0], (list, tuple)) else sp
            gt = [float(gt[0]), float(gt[1])]
            blue_spans = [s for q, s in all_spans_map.items()
                          if abs(s[0] - gt[0]) > 0.1 or abs(s[1] - gt[1]) > 0.1]

            colored = []
            for i, pil_frame in enumerate(frames):
                t = i / fps if fps > 0 else i * (a["duration"] / N)
                frame = pil_frame.convert("RGB")
                for bs in blue_spans:
                    if bs[0] <= t <= bs[1]:
                        frame = _add_border(frame, "blue")
                if gt[0] <= t <= gt[1]:
                    frame = _add_border(frame, "red")
                colored.append(np.array(frame))

            writer = imageio.get_writer(out_path, fps=fps, codec="libx264")
            for f_arr in colored:
                writer.append_data(f_arr)
            writer.close()
            done += 1

        if done % 100 == 0 or done >= len(annos):
            print(f"  [{done}/{len(annos)}] videos cached")

    print(f"Done. {done} videos -> {CACHE_DIR}")


if __name__ == "__main__":
    main()
