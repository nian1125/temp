"""Prepare ZwZ-style R2I training data for temporal grounding.

ZwZ approach: Student trains on MARKED full video frames → RL (tIoU reward).
At inference, marks are removed. The student internalizes temporal perception.

Unlike OPD (where teacher sees marks, student sees clean video),
in ZwZ-style training the student DIRECTLY sees the marks during training.

Output parquet columns (EasyR1-compatible):
  - images: list of marked frame paths (ZwZ uses images, not video)
  - problem: temporal grounding question
  - answer: GT timestamp answer
  - data_source: dataset identifier
  - ability: "temporal_grounding"
  - extra_info: metadata (query, span, video_id, duration)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.data_utils import load_charades_annos
from tools.temporal_crop import load_video_frames
from tools.frame_marker import mark_frames


PROMPT_TEMPLATE = (
    "Please find the visual event described by the sentence '{query}', "
    "determining its starting and ending times. The format should be: "
    "'The event happens in <answer><start time> - <end time> seconds</answer>'."
)


def process_single_sample(args_tuple):
    """Process one video sample: extract frames, apply marks, save."""
    anno, output_dir, marking_strategy, num_frames, fps = args_tuple

    video_path = anno["video_path"]
    video_id = anno.get("video_id", Path(video_path).stem)
    query = anno["query"]
    s, e = anno["span"]
    duration = anno.get("duration", 0.0)

    if not os.path.exists(video_path):
        return None

    try:
        frames, timestamps = load_video_frames(
            video_path, num_frames=num_frames,
            backend="decord", start_sec=0, end_sec=None,
        )
    except Exception:
        return None

    if not frames:
        return None

    marked_result = mark_frames(
        frames, timestamps, marking_strategy,
        gt_start=s, gt_end=e,
    )
    marked_frames = marked_result["frames"]

    sample_dir = os.path.join(output_dir, video_id, f"{s:.1f}_{e:.1f}")
    os.makedirs(sample_dir, exist_ok=True)

    frame_paths = []
    for i, frame in enumerate(marked_frames):
        frame_path = os.path.join(sample_dir, f"frame_{i:04d}.jpg")
        frame.save(frame_path, quality=85)
        frame_paths.append(frame_path)

    answer = f"The event happens in <answer>{s:.1f} - {e:.1f} seconds</answer>."

    return {
        "images": frame_paths,
        "problem": PROMPT_TEMPLATE.format(query=query),
        "answer": answer,
        "data_source": "temporal_zwz",
        "ability": "temporal_grounding",
        "extra_info": json.dumps({
            "query": query,
            "span": [s, e],
            "video_id": video_id,
            "duration": duration,
            "marking_strategy": marking_strategy,
            "video_path": video_path,
        }),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="charades")
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--durations_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_frames_dir", type=str, required=True,
                        help="Directory to save marked video frames")
    parser.add_argument("--marking_strategy", type=str, default="numberit",
                        choices=["numberit", "red_border", "t2sgrid", "timemarker",
                                 "numberit+red_border", "crop+numberit"])
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading {args.dataset} annotations...")
    annos = load_charades_annos(
        args.anno_path, args.video_dir,
        durations_path=args.durations_path,
    )
    if args.max_samples:
        annos = annos[:args.max_samples]
    print(f"Loaded {len(annos)} annotations")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    os.makedirs(args.output_frames_dir, exist_ok=True)

    task_args = [
        (anno, args.output_frames_dir, args.marking_strategy,
         args.num_frames, args.fps)
        for anno in annos
    ]

    results = []
    print(f"Processing {len(task_args)} samples with {args.num_workers} workers...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_sample, ta): i
                   for i, ta in enumerate(task_args)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result)

    print(f"Successfully processed {len(results)}/{len(annos)} samples")

    dataset = Dataset.from_dict({
        "images": [r["images"] for r in results],
        "problem": [r["problem"] for r in results],
        "answer": [r["answer"] for r in results],
        "data_source": [r["data_source"] for r in results],
        "ability": [r["ability"] for r in results],
        "extra_info": [r["extra_info"] for r in results],
    })
    dataset.to_parquet(args.output_path)
    print(f"Saved parquet to {args.output_path}")


if __name__ == "__main__":
    main()
