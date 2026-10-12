"""Convert TVG datasets (Charades-STA, ActivityNet) to Vision-OPD parquet format.

Produces parquet files compatible with Vision-OPD's RLHFDataset pipeline.

Columns:
  data_source  - str, e.g. "temporal_zwz_tvg"
  prompt       - list[dict], chat messages with <video> placeholder
  images       - list[dict], student video (empty for video-first mode)
  videos       - list[dict], student's full video reference
  bbox_images  - list[dict], teacher's marked clip frames (paths)
  ability      - str, "temporal_grounding"
  reward_model - dict, ground truth span + duration
  extra_info   - dict, query, span, video_id, etc.

Usage:
  python prepare_video_data.py \\
    --dataset charades \\
    --anno_path /path/to/train.json \\
    --video_dir /path/to/videos/ \\
    --output_path /path/to/output.parquet \\
    [--preprocess_teacher --teacher_output_dir /path/to/marked_clips/]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Add parent dir to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.data_utils import load_charades_annos, load_anet_annos
from tools.temporal_crop import extract_clip_frames
from tools.frame_marker import mark_frames


# TimeLens-style prompt template
GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{query}', "
    "determining its starting and ending times. The format should be: "
    "'The event happens in <start time> - <end time> seconds'."
)

# GRPO-style prompt template (with reasoning)
GRPO_PROMPT = (
    'To accurately pinpoint the event "{query}" in the video, '
    "determine the precise time period of the event.\n\n"
    "Output your thought process within the <think> </think> tags, "
    "including analysis with either specific timestamps (xx.xx) or "
    "time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.\n\n"
    "Then, provide the start and end times (in seconds, precise to two "
    'decimal places) in the format "start time to end time" within the '
    '<answer> </answer> tags. For example: "12.54 to 17.83".'
)


def build_prompt_messages(query: str, prompt_style: str = "timelens") -> list[dict]:
    """Build student chat messages (uses <video> placeholder)."""
    if prompt_style == "timelens":
        template = GROUNDING_PROMPT
    elif prompt_style == "grpo":
        template = GRPO_PROMPT
    else:
        template = GROUNDING_PROMPT

    content = "<video>\n" + template.format(query=query)
    return [{"role": "user", "content": content}]


def build_teacher_prompt_messages(
    query: str,
    num_frames: int,
    prompt_style: str = "timelens",
) -> list[dict]:
    """Build teacher chat messages (uses <image> placeholders for marked clip frames).

    The teacher sees marked clip frames as images instead of the full video.
    Number of <image> placeholders must match len(bbox_images).
    """
    if prompt_style == "timelens":
        template = GROUNDING_PROMPT
    elif prompt_style == "grpo":
        template = GRPO_PROMPT
    else:
        template = GROUNDING_PROMPT

    image_tags = "<image>\n" * num_frames
    content = image_tags + template.format(query=query)
    return [{"role": "user", "content": content}]


def build_answer(span: list[float], prompt_style: str = "timelens") -> str:
    """Build expected answer string."""
    s, e = span
    if prompt_style == "timelens":
        return f"The event happens in {s:.1f} - {e:.1f} seconds."
    elif prompt_style == "grpo":
        return f"<think>The event occurs from {s:.2f} to {e:.2f} seconds.</think>\n<answer>{s:.2f} to {e:.2f}</answer>"
    return f"The event happens in {s:.1f} - {e:.1f} seconds."


def build_record(
    anno: dict,
    prompt_style: str = "timelens",
    teacher_frame_dir: str | None = None,
    marking_strategy: str = "numberit",
    num_teacher_frames: int = 16,
    video_fps: float = 2.0,
    video_max_frames: int = 128,
) -> dict:
    """Build a single Vision-OPD compatible record.

    Args:
        anno: Annotation dict from data_utils loaders.
        prompt_style: "timelens" or "grpo".
        teacher_frame_dir: If not None, pre-extracted marked frames directory.
        marking_strategy: Marking strategy for teacher frames.
        num_teacher_frames: Number of frames to extract for teacher.
        video_fps: FPS for student video sampling.
        video_max_frames: Max frames for student video.

    Returns:
        Dict with parquet column values.
    """
    query = anno["query"]
    span = anno["span"]
    video_path = anno["video_path"]
    duration = anno.get("duration") or 0.0
    video_id = anno.get("video_id", Path(video_path).stem)

    # Build prompt
    prompt = build_prompt_messages(query, prompt_style)
    answer = build_answer(span, prompt_style)

    # Student video reference
    video_ref = {
        "video": video_path,
        "fps": video_fps,
        "max_frames": video_max_frames,
    }

    # Teacher marked clip frames
    bbox_images = []
    if teacher_frame_dir:
        sample_dir = os.path.join(teacher_frame_dir, video_id, f"{span[0]:.1f}_{span[1]:.1f}")
        if os.path.isdir(sample_dir):
            frame_files = sorted(
                f for f in os.listdir(sample_dir)
                if f.endswith((".jpg", ".png"))
            )
            bbox_images = [
                {"path": os.path.join(sample_dir, f)} for f in frame_files
            ]

    # Teacher prompt: uses <image> placeholders matching bbox_images count
    # If no teacher frames yet, use num_teacher_frames as placeholder count
    n_teacher = len(bbox_images) if bbox_images else num_teacher_frames
    teacher_prompt = build_teacher_prompt_messages(query, n_teacher, prompt_style)

    record = {
        "data_source": "temporal_zwz_tvg",
        "prompt": prompt,
        "teacher_prompt": teacher_prompt,
        "images": [],
        "videos": [video_ref],
        "bbox_images": bbox_images,
        "ability": "temporal_grounding",
        "reward_model": {
            "style": "iou",
            "ground_truth": {
                "span": span,
                "duration": duration,
                "answer": answer,
            },
        },
        "extra_info": {
            "query": query,
            "span": span,
            "duration": duration,
            "video_id": video_id,
            "video_path": video_path,
            "marking_strategy": marking_strategy,
        },
    }
    return record


def preprocess_teacher_frames(
    anno: dict,
    output_dir: str,
    marking_strategy: str = "numberit",
    num_frames: int = 16,
    backend: str = "decord",
    font_size: int = 40,
) -> bool:
    """Pre-extract and save marked teacher clip frames."""
    video_path = anno["video_path"]
    span = anno["span"]
    video_id = anno.get("video_id", Path(video_path).stem)

    sample_dir = os.path.join(output_dir, video_id, f"{span[0]:.1f}_{span[1]:.1f}")
    if os.path.isdir(sample_dir) and len(os.listdir(sample_dir)) > 0:
        return True

    os.makedirs(sample_dir, exist_ok=True)

    try:
        frames, timestamps = extract_clip_frames(
            video_path, span[0], span[1],
            num_frames=num_frames, backend=backend,
        )
        if not frames:
            return False

        marked = mark_frames(
            frames, timestamps,
            strategy=marking_strategy,
            gt_start=span[0], gt_end=span[1],
            font_size=font_size,
        )

        for i, frame in enumerate(marked["frames"]):
            frame_path = os.path.join(sample_dir, f"frame_{i:04d}.jpg")
            frame.save(frame_path, "JPEG", quality=90)

        return True
    except Exception as e:
        print(f"Error processing {video_id} [{span[0]}-{span[1]}]: {e}")
        return False


def filter_valid_annos(annos: list[dict]) -> list[dict]:
    """Filter annotations with valid video paths and spans."""
    valid = []
    for anno in annos:
        if not os.path.exists(anno["video_path"]):
            continue
        span = anno["span"]
        if span[0] >= span[1]:
            continue
        if anno.get("duration") and span[1] > anno["duration"] + 1.0:
            continue
        valid.append(anno)
    return valid


def main():
    parser = argparse.ArgumentParser(description="Convert TVG datasets to Vision-OPD parquet")
    parser.add_argument("--dataset", type=str, required=True, choices=["charades", "anet"])
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prompt_style", type=str, default="timelens", choices=["timelens", "grpo"])
    parser.add_argument("--marking_strategy", type=str, default="numberit",
                        choices=["numberit", "red_border", "t2sgrid", "numberit+red_border", "crop+numberit"])
    parser.add_argument("--num_teacher_frames", type=int, default=16)
    parser.add_argument("--video_fps", type=float, default=2.0)
    parser.add_argument("--video_max_frames", type=int, default=128)
    parser.add_argument("--video_backend", type=str, default="decord", choices=["decord", "opencv"])

    parser.add_argument("--preprocess_teacher", action="store_true",
                        help="Pre-extract and mark teacher clip frames")
    parser.add_argument("--teacher_output_dir", type=str, default=None,
                        help="Directory to save marked teacher frames")
    parser.add_argument("--font_size", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    args = parser.parse_args()

    # Load annotations
    print(f"Loading {args.dataset} annotations from {args.anno_path}...")
    if args.dataset == "charades":
        annos = load_charades_annos(args.anno_path, args.video_dir, args.split)
    else:
        annos = load_anet_annos(args.anno_path, args.video_dir, args.split)

    print(f"  Loaded {len(annos)} annotations")

    # Filter valid
    annos = filter_valid_annos(annos)
    print(f"  After filtering: {len(annos)} valid annotations")

    if args.max_samples:
        annos = annos[:args.max_samples]
        print(f"  Truncated to {len(annos)} samples")

    # Pre-process teacher frames if requested
    teacher_dir = args.teacher_output_dir
    if args.preprocess_teacher:
        if teacher_dir is None:
            teacher_dir = os.path.join(
                os.path.dirname(args.output_path),
                f"teacher_frames_{args.marking_strategy}",
            )
        os.makedirs(teacher_dir, exist_ok=True)
        print(f"Pre-extracting teacher frames to {teacher_dir}...")

        success = 0
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    preprocess_teacher_frames,
                    anno, teacher_dir, args.marking_strategy,
                    args.num_teacher_frames, args.video_backend, args.font_size,
                )
                for anno in annos
            ]
            for future in tqdm(futures, total=len(futures), desc="Extracting"):
                if future.result():
                    success += 1

        print(f"  Successfully processed {success}/{len(annos)} samples")

    # Build records
    print("Building parquet records...")
    records = []
    for anno in tqdm(annos, desc="Building records"):
        record = build_record(
            anno,
            prompt_style=args.prompt_style,
            teacher_frame_dir=teacher_dir,
            marking_strategy=args.marking_strategy,
            num_teacher_frames=args.num_teacher_frames,
            video_fps=args.video_fps,
            video_max_frames=args.video_max_frames,
        )
        records.append(record)

    # Save as parquet
    print(f"Saving {len(records)} records to {args.output_path}...")
    from datasets import Dataset
    ds = Dataset.from_list(records)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    ds.to_parquet(args.output_path)
    print(f"Done! Saved to {args.output_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Records: {len(records)}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Marking strategy: {args.marking_strategy}")
    print(f"Teacher frames pre-processed: {args.preprocess_teacher}")
    if teacher_dir:
        print(f"Teacher frames dir: {teacher_dir}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
