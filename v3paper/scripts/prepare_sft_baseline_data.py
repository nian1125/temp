"""Prepare SFT baseline data (for comparison only, NOT the main method).

NOTE: The main Temporal-ZwZ method uses RL (ZwZ-style or OPD-style), not SFT.
This script prepares data for the SFT baseline comparison.

Two modes:
  1. GT mode (default): Use ground-truth answers directly.
  2. Teacher mode (--teacher_model): Run teacher inference on marked clip frames,
     collect predictions as training targets.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.data_utils import load_charades_annos
from tools.temporal_crop import extract_clip_frames
from tools.frame_marker import mark_frames


def generate_gt_distill_data(
    anno_path: str,
    video_dir: str,
    output_path: str,
    marking_strategy: str,
    durations_path: str | None = None,
    num_teacher_frames: int = 16,
):
    """Generate SFT data using GT answers (no teacher inference needed)."""
    annos = load_charades_annos(anno_path, video_dir, durations_path=durations_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for anno in annos:
            if not os.path.exists(anno["video_path"]):
                continue
            s, e = anno["span"]
            record = {
                "video_path": anno["video_path"],
                "video_id": anno.get("video_id", ""),
                "query": anno["query"],
                "answer": f"The event happens in {s:.1f} - {e:.1f} seconds.",
                "span": anno["span"],
                "duration": anno.get("duration", 0.0),
                "marking_strategy": marking_strategy,
                "source": "gt_distill",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved {count} GT-distill samples to {output_path}")


def generate_teacher_distill_data(
    anno_path: str,
    video_dir: str,
    output_path: str,
    teacher_model_path: str,
    marking_strategy: str,
    durations_path: str | None = None,
    num_teacher_frames: int = 16,
    batch_size: int = 8,
):
    """Generate SFT data by running teacher inference on marked clip frames."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    annos = load_charades_annos(anno_path, video_dir, durations_path=durations_path)

    print(f"Loading teacher model from {teacher_model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        teacher_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(teacher_model_path, trust_remote_code=True)

    PROMPT_TEMPLATE = (
        "Please find the visual event described by the sentence '{query}', "
        "determining its starting and ending times. The format should be: "
        "'The event happens in <start time> - <end time> seconds'."
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0

    with open(output_path, "w") as f:
        for anno in annos:
            if not os.path.exists(anno["video_path"]):
                continue

            s_gt, e_gt = anno["span"]
            duration = anno.get("duration", 0.0)

            try:
                frames, timestamps = extract_clip_frames(
                    anno["video_path"], s_gt, e_gt,
                    num_frames=num_teacher_frames,
                )
            except Exception:
                continue

            marked = mark_frames(
                frames, timestamps, marking_strategy,
                gt_start=s_gt, gt_end=e_gt,
            )
            teacher_frames = marked["frames"]

            content = []
            for frame in teacher_frames:
                content.append({"type": "image", "image": frame})
            content.append({
                "type": "text",
                "text": PROMPT_TEMPLATE.format(query=anno["query"]),
            })

            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=256)

            input_len = inputs["input_ids"].shape[1]
            teacher_response = processor.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )

            record = {
                "video_path": anno["video_path"],
                "video_id": anno.get("video_id", ""),
                "query": anno["query"],
                "answer": teacher_response,
                "gt_answer": f"The event happens in {s_gt:.1f} - {e_gt:.1f} seconds.",
                "span": anno["span"],
                "duration": duration,
                "marking_strategy": marking_strategy,
                "source": "teacher_distill",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count % 100 == 0:
                print(f"  Processed {count} samples...")

    print(f"Saved {count} teacher-distill samples to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--durations_path", type=str, default=None)
    parser.add_argument("--marking_strategy", type=str, default="numberit")
    parser.add_argument("--num_teacher_frames", type=int, default=16)
    parser.add_argument("--teacher_model", type=str, default=None,
                        help="Path to teacher model. If not set, uses GT answers.")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if args.teacher_model:
        generate_teacher_distill_data(
            args.anno_path, args.video_dir, args.output_path,
            args.teacher_model, args.marking_strategy,
            args.durations_path, args.num_teacher_frames, args.batch_size,
        )
    else:
        generate_gt_distill_data(
            args.anno_path, args.video_dir, args.output_path,
            args.marking_strategy, args.durations_path, args.num_teacher_frames,
        )


if __name__ == "__main__":
    main()
