"""Unified data preparation for both image and video modalities.

Generates training parquet that contains both:
- Video temporal grounding samples (from Charades-STA / ActivityNet)
- Image fine-grained perception samples (from Vision-OPD-6K / ZwZ-RL-VQA)

Both modalities go through the same marking pipeline (NumberIT, etc.)
and are output in a unified format for joint training.

Output parquet columns:
  - images: list of marked frame/tile paths
  - problem: VQA question
  - answer: ground truth answer
  - data_source: "temporal_grounding" | "image_perception"
  - ability: task type
  - extra_info: JSON metadata
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.data_utils import load_charades_annos
from tools.temporal_crop import load_video_frames
from tools.frame_marker import mark_frames
from tools.image_tile import split_image_to_tiles, compute_tile_grid, is_high_res


# ---- Video temporal grounding ----

TEMPORAL_PROMPT = (
    "Please find the visual event described by the sentence '{query}', "
    "determining its starting and ending times. The format should be: "
    "'The event happens in <answer><start time> - <end time> seconds</answer>'."
)


def process_video_sample(args_tuple):
    """Process one video sample with marks for ZwZ-style training."""
    anno, output_dir, marking_strategy, num_frames = args_tuple

    video_path = anno["video_path"]
    video_id = anno.get("video_id", Path(video_path).stem)
    query = anno["query"]
    s, e = anno["span"]
    duration = anno.get("duration", 0.0)

    if not os.path.exists(video_path):
        return None

    try:
        frames, timestamps = load_video_frames(
            video_path, num_frames=num_frames, backend="decord",
        )
    except Exception:
        return None

    if not frames:
        return None

    marked = mark_frames(frames, timestamps, marking_strategy, gt_start=s, gt_end=e)

    sample_dir = os.path.join(output_dir, "video", video_id, f"{s:.1f}_{e:.1f}")
    os.makedirs(sample_dir, exist_ok=True)

    frame_paths = []
    for i, frame in enumerate(marked["frames"]):
        fp = os.path.join(sample_dir, f"frame_{i:04d}.jpg")
        frame.save(fp, quality=85)
        frame_paths.append(fp)

    return {
        "images": frame_paths,
        "problem": TEMPORAL_PROMPT.format(query=query),
        "answer": f"The event happens in <answer>{s:.1f} - {e:.1f} seconds</answer>.",
        "data_source": "temporal_grounding",
        "ability": "temporal_grounding",
        "extra_info": json.dumps({
            "query": query, "span": [s, e],
            "video_id": video_id, "duration": duration,
            "marking_strategy": marking_strategy,
        }),
    }


# ---- Image fine-grained perception ----

IMAGE_PROMPT_TEMPLATE = "{question}"


def process_image_sample(args_tuple):
    """Process one image sample: tile if high-res → mark → save."""
    record, output_dir, marking_strategy, max_tiles = args_tuple

    images = record.get("images", [])
    if not images:
        return None

    image_path = images[0] if isinstance(images[0], str) else None
    if image_path is None:
        return None

    if not os.path.exists(image_path):
        return None

    from PIL import Image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    w, h = image.size
    bbox = record.get("bbox")

    if is_high_res(image):
        cols, rows = compute_tile_grid(w, h, max_tiles)
        tiles = split_image_to_tiles(image, cols, rows)
    else:
        tiles = [image]
        cols, rows = 1, 1

    timestamps = [float(i) for i in range(len(tiles))]
    gt_start, gt_end = None, None

    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        tile_w = w // cols
        tile_h = h // rows
        sc = int(x1 // tile_w)
        sr = int(y1 // tile_h)
        ec = min(int(x2 // tile_w), cols - 1)
        er = min(int(y2 // tile_h), rows - 1)
        gt_start = float(sr * cols + sc)
        gt_end = float(er * cols + ec)

    marked = mark_frames(tiles, timestamps, marking_strategy,
                         gt_start=gt_start, gt_end=gt_end)

    sample_id = Path(image_path).stem
    sample_dir = os.path.join(output_dir, "image", sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    frame_paths = []
    for i, frame in enumerate(marked["frames"]):
        fp = os.path.join(sample_dir, f"tile_{i:04d}.jpg")
        frame.save(fp, quality=85)
        frame_paths.append(fp)

    return {
        "images": frame_paths,
        "problem": record.get("problem", ""),
        "answer": record.get("answer", ""),
        "data_source": "image_perception",
        "ability": record.get("ability", "fine_grained_perception"),
        "extra_info": json.dumps({
            "image_path": image_path,
            "bbox": bbox,
            "grid": [cols, rows],
            "marking_strategy": marking_strategy,
        }),
    }


def prepare_video_data(args, output_dir, marking_strategy):
    """Prepare video temporal grounding samples."""
    if not args.video_anno_path:
        return []

    annos = load_charades_annos(
        args.video_anno_path, args.video_dir,
        durations_path=args.durations_path,
    )
    if args.max_video_samples:
        annos = annos[:args.max_video_samples]

    task_args = [
        (anno, output_dir, marking_strategy, args.num_video_frames)
        for anno in annos
    ]

    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_video_sample, ta): i
                   for i, ta in enumerate(task_args)}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Video samples"):
            result = future.result()
            if result:
                results.append(result)

    return results


def prepare_image_data(args, output_dir, marking_strategy):
    """Prepare image perception samples from HuggingFace datasets."""
    if not args.image_dataset:
        return []

    print(f"Loading image dataset: {args.image_dataset}")
    ds = load_dataset(args.image_dataset, split="train")
    records = list(ds)
    if args.max_image_samples:
        records = records[:args.max_image_samples]

    task_args = [
        (record, output_dir, marking_strategy, args.max_tiles)
        for record in records
    ]

    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_image_sample, ta): i
                   for i, ta in enumerate(task_args)}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Image samples"):
            result = future.result()
            if result:
                results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    # Video args
    parser.add_argument("--video_anno_path", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--durations_path", type=str, default=None)
    parser.add_argument("--num_video_frames", type=int, default=32)
    parser.add_argument("--max_video_samples", type=int, default=None)
    # Image args
    parser.add_argument("--image_dataset", type=str, default=None,
                        help="HuggingFace dataset name, e.g. yuanqianhao/Vision-OPD-6K")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--max_tiles", type=int, default=16)
    parser.add_argument("--max_image_samples", type=int, default=None)
    # Common args
    parser.add_argument("--marking_strategy", type=str, default="numberit")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_frames_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    os.makedirs(args.output_frames_dir, exist_ok=True)

    all_results = []

    video_results = prepare_video_data(args, args.output_frames_dir, args.marking_strategy)
    print(f"Video samples: {len(video_results)}")
    all_results.extend(video_results)

    image_results = prepare_image_data(args, args.output_frames_dir, args.marking_strategy)
    print(f"Image samples: {len(image_results)}")
    all_results.extend(image_results)

    print(f"Total samples: {len(all_results)}")

    dataset = Dataset.from_dict({
        k: [r[k] for r in all_results]
        for k in ["images", "problem", "answer", "data_source", "ability", "extra_info"]
    })
    dataset.to_parquet(args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
