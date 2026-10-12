"""Visualize teacher marking strategies on sample video frames.

Usage:
    python scripts/viz_teacher_strategies.py \
        --video_path /cpfs01/user/chenzhaoyu.czy/data/charades/raw_480p/Charades_v1_480/3MSZA.mp4 \
        --gt_start 25 --gt_end 30 \
        --output_dir /cpfs01/user/chenzhaoyu.czy/code/Temporal-ZwZ/viz_output \
        --num_frames 16
"""
import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def load_frames(video_path, num_frames=16):
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    fps = float(vr.get_avg_fps())
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = [idx / fps for idx in indices]
    return [Image.fromarray(f) for f in frames], timestamps, total / fps


def _get_font(size=28):
    for path in [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def add_red_border(frame, border_width=8):
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    w, h = frame.size
    for i in range(border_width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline="red")
    return frame


def add_text_banner(frame, text="TARGET EVENT", font_size=28):
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    font = _get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, _ = frame.size
    pad = 4
    bar_h = th + 2 * pad
    draw.rectangle([0, 0, w, bar_h], fill="red")
    draw.text(((w - tw) // 2, pad), text, font=font, fill="white")
    return frame


def add_frame_number(frame, number, font_size=40):
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    font = _get_font(font_size)
    text = str(number)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = frame.size
    draw.text((w - tw - 10, h - th - 10), text, font=font, fill="red")
    return frame


def add_timestamp_label(frame, ts, duration):
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    font = _get_font(16)
    text = f"{ts:.1f}s / {duration:.1f}s"
    draw.text((5, 5), text, font=font, fill="yellow")
    return frame


STRATEGIES = {
    "gt_text": lambda f, i, in_gt: add_text_banner(add_red_border(f)) if in_gt else f,
    "numberit+red_border": lambda f, i, in_gt: add_red_border(add_frame_number(f, i)) if in_gt else add_frame_number(f, i),
    "gt_only_numberit+red_border": lambda f, i, in_gt: add_red_border(add_frame_number(f, i)) if in_gt else f,
    "clean": lambda f, i, in_gt: f,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--gt_start", type=float, required=True)
    parser.add_argument("--gt_end", type=float, required=True)
    parser.add_argument("--output_dir", default="viz_output")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES.keys()))
    args = parser.parse_args()

    frames, timestamps, duration = load_frames(args.video_path, args.num_frames)
    print(f"Video: {args.video_path}")
    print(f"Duration: {duration:.1f}s, GT span: [{args.gt_start}, {args.gt_end}]")
    print(f"Sampled {len(frames)} frames")

    for strategy in args.strategies:
        if strategy not in STRATEGIES:
            print(f"[SKIP] Unknown strategy: {strategy}")
            continue
        out_dir = os.path.join(args.output_dir, strategy)
        os.makedirs(out_dir, exist_ok=True)

        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            in_gt = args.gt_start <= ts <= args.gt_end
            marked = STRATEGIES[strategy](frame, i, in_gt)
            marked = add_timestamp_label(marked, ts, duration)
            tag = "GT" if in_gt else ""
            out_path = os.path.join(out_dir, f"frame_{i:03d}_{ts:.1f}s{'_GT' if in_gt else ''}.jpg")
            marked.save(out_path)

        print(f"[{strategy}] Saved {len(frames)} frames to {out_dir}/")


if __name__ == "__main__":
    main()
