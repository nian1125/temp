"""Visualize marking strategies on a video (no GPU required).

Applies all 6 marking strategies to a video and saves output frames.
Useful for verifying marks look correct before running inference.

Usage:
  python scripts/visualize_marks.py \
      --video_path data/charades/videos/XXXXX.mp4 \
      --output_dir results/mark_samples/ \
      --num_frames 16 \
      --gt_start 5.0 --gt_end 12.0
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.temporal_crop import load_video_frames, extract_clip_frames, get_video_duration
from tools.frame_marker import mark_frames

ALL_STRATEGIES = ["numberit", "red_border", "t2sgrid", "timemarker", "numberit+red_border", "crop+numberit"]


def main():
    parser = argparse.ArgumentParser(description="Visualize marking strategies")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/mark_samples")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--gt_start", type=float, default=None,
                        help="GT segment start (seconds). Auto-set to 30-60%% of duration if not given.")
    parser.add_argument("--gt_end", type=float, default=None,
                        help="GT segment end (seconds). Auto-set to 30-60%% of duration if not given.")
    parser.add_argument("--strategies", type=str, default="all",
                        help="Comma-separated strategies or 'all'")
    args = parser.parse_args()

    if args.strategies == "all":
        strategies = ALL_STRATEGIES
    else:
        strategies = [s.strip() for s in args.strategies.split(",")]

    duration = get_video_duration(args.video_path)
    print(f"Video: {args.video_path}")
    print(f"Duration: {duration:.1f}s")

    gt_start = args.gt_start if args.gt_start is not None else duration * 0.3
    gt_end = args.gt_end if args.gt_end is not None else duration * 0.6
    print(f"GT segment: [{gt_start:.1f}, {gt_end:.1f}]")

    frames, timestamps = load_video_frames(args.video_path, num_frames=args.num_frames)
    print(f"Extracted {len(frames)} frames: {[f'{t:.1f}s' for t in timestamps]}")

    vid = Path(args.video_path).stem
    os.makedirs(args.output_dir, exist_ok=True)

    clean_dir = os.path.join(args.output_dir, f"{vid}_clean")
    os.makedirs(clean_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(clean_dir, f"frame_{i:03d}.jpg"))
    print(f"\nSaved clean frames → {clean_dir}/")

    for strategy in strategies:
        print(f"\nApplying: {strategy}")
        strat_dir = os.path.join(args.output_dir, f"{vid}_{strategy.replace('+', '_')}")
        os.makedirs(strat_dir, exist_ok=True)

        if strategy == "crop+numberit":
            clip_frames, clip_ts = extract_clip_frames(
                args.video_path, gt_start, gt_end, num_frames=args.num_frames
            )
            result = mark_frames(clip_frames, clip_ts, strategy=strategy)
        else:
            result = mark_frames(
                frames, timestamps, strategy=strategy,
                gt_start=gt_start, gt_end=gt_end,
            )

        marked = result["frames"]
        tokens = result["tokens"]

        for i, frame in enumerate(marked):
            frame.save(os.path.join(strat_dir, f"frame_{i:03d}.jpg"))

        if tokens:
            with open(os.path.join(strat_dir, "tokens.txt"), "w") as f:
                for t in tokens:
                    f.write(t + "\n")
            print(f"  Tokens: {tokens[:5]}{'...' if len(tokens) > 5 else ''}")

        print(f"  Saved {len(marked)} frames → {strat_dir}/")

    print(f"\nDone! Check {args.output_dir}/ for results.")


if __name__ == "__main__":
    main()
