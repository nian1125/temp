"""Zero-Shot TVG Test: Compare clean video vs marked-video strategies.

Validates the teacher-student premise: do GT-dependent visual marks help
a VLM produce more accurate temporal grounding predictions?

All strategies feed VIDEO input (not images) so Qwen3-VL's built-in
timemarker is preserved. Marks are burned into the video frames.

Strategies:
  clean               — original video, no marks (baseline)
  red_border          — GT-segment frames get red border
  numberit            — all frames get index number overlay
  numberit+red_border — index overlay + red border on GT frames
  crop+numberit       — crop to GT segment video, then number all frames

Usage:
  python scripts/zero_shot_tvg_test.py \
      --model_path /path/to/Qwen3-VL-8B-Instruct \
      --dataset timelens_charades \
      --anno_path /path/to/charades-timelens.json \
      --video_dir /path/to/videos/ \
      --strategies clean,red_border,numberit+red_border,crop+numberit \
      --max_samples 200 \
      --output_dir results/zero_shot_tvg/
"""

import argparse
import gc
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.data_utils import (
    load_charades_annos,
    load_anet_annos,
    parse_timestamp_output,
    compute_temporal_iou,
    evaluate_predictions,
)
from tools.frame_marker import annotate_frame_number, add_red_border

ALL_STRATEGIES = ["clean", "red_border", "numberit", "numberit+red_border", "crop+numberit"]

# ---------------------------------------------------------------------------
# Prompt templates (English, video-oriented)
# ---------------------------------------------------------------------------

GROUNDER_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

PROMPTS = {
    "clean": GROUNDER_PROMPT,
    "red_border": (
        "Some frames in this video are highlighted with a red border — "
        "these frames fall within the target event's time range. "
        + GROUNDER_PROMPT
    ),
    "numberit": (
        "Each frame in this video has its sequential index number overlaid. "
        + GROUNDER_PROMPT
    ),
    "numberit+red_border": (
        "Each frame in this video has its index number overlaid. "
        "Frames with a red border fall within the target event's time range. "
        + GROUNDER_PROMPT
    ),
    "crop+numberit": (
        "This is a clip extracted from the original video, spanning "
        "approximately {crop_start:.1f}s to {crop_end:.1f}s. Each frame is numbered. "
        + GROUNDER_PROMPT
    ),
}


# ---------------------------------------------------------------------------
# Video creation utilities
# ---------------------------------------------------------------------------

def read_video_frames(video_path):
    """Read ALL frames from a video using decord. Returns (list[PIL], fps)."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    frames = []
    for i in range(len(vr)):
        arr = vr[i].asnumpy()
        frames.append(Image.fromarray(arr))
    return frames, fps


def write_video(frames, fps, output_path):
    """Write PIL frames to a video file via cv2."""
    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def create_marked_video(video_path, strategy, gt_start, gt_end, tmp_dir):
    """Read video, apply marks to ALL frames, write to temp file.

    Returns (temp_video_path, duration).
    """
    frames, fps = read_video_frames(video_path)
    n = len(frames)
    duration = n / fps if fps > 0 else 0.0

    marked = []
    for i, frame in enumerate(frames):
        ts = i / fps if fps > 0 else 0.0
        in_gt = gt_start <= ts <= gt_end

        if strategy == "red_border":
            marked.append(add_red_border(frame) if in_gt else frame)

        elif strategy == "numberit":
            marked.append(annotate_frame_number(frame, i))

        elif strategy == "numberit+red_border":
            f = annotate_frame_number(frame, i)
            if in_gt:
                f = add_red_border(f)
            marked.append(f)

        else:
            marked.append(frame)

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(tmp_dir, f"{strategy}_{base}.mp4")
    write_video(marked, fps, out_path)
    return out_path, duration


def create_cropped_video(video_path, gt_start, gt_end, tmp_dir):
    """Create a video of the GT segment with numbered frames.

    Returns (temp_video_path, full_duration, crop_start, crop_end).
    """
    frames, fps = read_video_frames(video_path)
    full_duration = len(frames) / fps if fps > 0 else 0.0

    start_idx = max(0, int(gt_start * fps))
    end_idx = min(len(frames), int(gt_end * fps) + 1)

    if start_idx >= end_idx:
        start_idx = max(0, end_idx - 1)

    segment = []
    for j, i in enumerate(range(start_idx, end_idx)):
        f = annotate_frame_number(frames[i], j)
        segment.append(f)

    actual_start = start_idx / fps if fps > 0 else gt_start
    actual_end = (end_idx - 1) / fps if fps > 0 else gt_end

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(tmp_dir, f"crop_{base}.mp4")
    write_video(segment, fps, out_path)
    return out_path, full_duration, actual_start, actual_end


# ---------------------------------------------------------------------------
# Build Qwen-VL messages
# ---------------------------------------------------------------------------

def build_video_messages(strategy, video_path, query, duration,
                         crop_start=None, crop_end=None,
                         fps=2, min_pixels=16384, total_pixels=3670016):
    """Build chat messages with video input."""
    if strategy == "crop+numberit":
        prompt = (
            f"This is a clip extracted from the original video, spanning "
            f"approximately {crop_start:.1f}s to {crop_end:.1f}s. Each frame is numbered. "
            + GROUNDER_PROMPT.format(query)
        )
    else:
        prompt = PROMPTS[strategy].format(query)

    content = [
        {
            "type": "video",
            "video": video_path,
            "fps": fps,
            "min_pixels": min_pixels,
            "total_pixels": total_pixels,
        },
        {"type": "text", "text": prompt},
    ]
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3-VL output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def load_existing_predictions(pred_file: str) -> dict:
    """Load predictions keyed by (video_id, query)."""
    existing = {}
    if os.path.exists(pred_file):
        with open(pred_file, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    key = (obj["video_id"], obj["query"])
                    existing[key] = obj
    return existing


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_strategy(
    strategy: str,
    model,
    processor,
    annos: list[dict],
    output_dir: str,
    tmp_dir: str,
    force: bool = False,
    no_think: bool = False,
    fps: int = 2,
    min_pixels: int = 16384,
    total_pixels: int = 3670016,
):
    """Run zero-shot TVG for one strategy across all annotations."""
    pred_file = os.path.join(output_dir, f"predictions_{strategy}.jsonl")

    existing = {} if force else load_existing_predictions(pred_file)
    if existing:
        print(f"  Resuming: {len(existing)} existing predictions found")

    preds_all = []
    gts_all = []
    results = []
    skipped = 0
    parse_fail = 0

    for anno in tqdm(annos, desc=strategy):
        vid = anno["video_id"]
        query = anno["query"]
        gt_span = anno["span"]
        duration = anno["duration"]
        key = (vid, query)

        if key in existing:
            obj = existing[key]
            parsed = parse_timestamp_output(obj.get("prediction", ""))
            pred = parsed if parsed else (0.0, 0.0)
            preds_all.append(pred)
            gts_all.append((gt_span[0], gt_span[1]))
            results.append(obj)
            skipped += 1
            continue

        if not os.path.exists(anno["video_path"]):
            continue

        # --- Prepare video input ---
        try:
            if strategy == "clean":
                input_video = anno["video_path"]
                vid_dur = duration
                messages = build_video_messages(
                    strategy, input_video, query, vid_dur,
                    fps=fps, min_pixels=min_pixels, total_pixels=total_pixels,
                )
            elif strategy == "crop+numberit":
                input_video, vid_dur, cs, ce = create_cropped_video(
                    anno["video_path"], gt_span[0], gt_span[1], tmp_dir,
                )
                messages = build_video_messages(
                    strategy, input_video, query, vid_dur,
                    crop_start=cs, crop_end=ce,
                    fps=fps, min_pixels=min_pixels, total_pixels=total_pixels,
                )
            else:
                input_video, vid_dur = create_marked_video(
                    anno["video_path"], strategy,
                    gt_span[0], gt_span[1], tmp_dir,
                )
                messages = build_video_messages(
                    strategy, input_video, query, vid_dur,
                    fps=fps, min_pixels=min_pixels, total_pixels=total_pixels,
                )
        except Exception as e:
            print(f"\n  Error preparing video for {vid}: {e}")
            continue

        # Append /no_think
        if no_think:
            for item in reversed(messages[0]["content"]):
                if item["type"] == "text":
                    item["text"] += " /no_think"
                    break

        # --- Inference ---
        try:
            from qwen_vl_utils import process_vision_info

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if videos:
                videos_list, video_metadatas = zip(*videos)
                videos_list = list(videos_list)
                video_metadatas = list(video_metadatas)
            else:
                videos_list = None
                video_metadatas = None

            inputs = processor(
                text=[text],
                images=images,
                videos=videos_list,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=512,
                    do_sample=False, temperature=None, top_p=None,
                )

            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                output_ids[0][input_len:], skip_special_tokens=True,
            )
        except Exception as e:
            print(f"\n  Inference error on {vid}: {e}")
            response = ""

        # Cleanup temp video
        if strategy != "clean" and os.path.exists(input_video):
            os.remove(input_video)

        response_clean = strip_thinking(response)
        parsed = parse_timestamp_output(response_clean)
        if parsed is None:
            pred = (0.0, 0.0)
            parse_fail += 1
        else:
            pred = parsed

        preds_all.append(pred)
        gts_all.append((gt_span[0], gt_span[1]))

        iou = compute_temporal_iou(pred, (gt_span[0], gt_span[1]))
        result = {
            "video_id": vid,
            "query": query,
            "gt_span": gt_span,
            "prediction": response_clean,
            "pred_span": list(pred),
            "iou": round(iou, 4),
            "strategy": strategy,
        }
        results.append(result)

        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    with open(pred_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not preds_all:
        print(f"  No predictions for {strategy}")
        return {"strategy": strategy, "error": "no predictions"}

    metrics = evaluate_predictions(preds_all, gts_all)
    metrics["strategy"] = strategy
    metrics["total"] = len(preds_all)
    metrics["parse_fail"] = parse_fail
    metrics["skipped_resume"] = skipped
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Shot TVG: clean video vs marked-video strategies",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="timelens_charades",
                        choices=["charades", "anet", "timelens_charades", "timelens_anet"])
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--durations_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--strategies", type=str,
                        default="clean,red_border,numberit+red_border,crop+numberit")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 = all samples")
    parser.add_argument("--output_dir", type=str, default="results/zero_shot_tvg/")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_think", action="store_true",
                        help="Disable Qwen3-VL thinking mode")
    parser.add_argument("--fps", type=int, default=2,
                        help="Video sampling fps (default: 2, matching TimeLens)")
    parser.add_argument("--min_tokens", type=int, default=16)
    parser.add_argument("--total_tokens", type=int, default=3584)
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]
    for s in strategies:
        if s not in ALL_STRATEGIES:
            print(f"Error: unknown strategy '{s}'. Available: {ALL_STRATEGIES}")
            sys.exit(1)

    # --- Load annotations ---
    print(f"Loading {args.dataset} annotations...")
    if args.dataset == "charades":
        annos = load_charades_annos(
            args.anno_path, args.video_dir, args.split, args.durations_path,
        )
    elif args.dataset == "anet":
        annos = load_anet_annos(args.anno_path, args.video_dir, args.split)
    elif args.dataset.startswith("timelens"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_tvg import load_timelens_annos
        annos = load_timelens_annos(args.anno_path, args.video_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.max_samples > 0:
        annos = annos[:args.max_samples]
    print(f"Loaded {len(annos)} annotations")

    # --- Load model ---
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )
    print("Model loaded.")

    # --- Temp dir for marked videos ---
    tmp_dir = tempfile.mkdtemp(prefix="tvg_marked_")
    print(f"Temp dir: {tmp_dir}")

    # --- Run each strategy ---
    all_metrics = []
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        t0 = time.time()
        downsample_rate = 32
        min_pixels = args.min_tokens * downsample_rate * downsample_rate
        total_pixels = args.total_tokens * downsample_rate * downsample_rate

        metrics = run_strategy(
            strategy=strategy,
            model=model,
            processor=processor,
            annos=annos,
            output_dir=args.output_dir,
            tmp_dir=tmp_dir,
            force=args.force,
            no_think=args.no_think,
            fps=args.fps,
            min_pixels=min_pixels,
            total_pixels=total_pixels,
        )
        elapsed = time.time() - t0
        metrics["time_sec"] = round(elapsed, 1)
        all_metrics.append(metrics)

        print(f"  mIoU:    {metrics.get('mIoU', 0):.4f}")
        print(f"  R1@0.3:  {metrics.get('R1@0.3', 0):.4f}")
        print(f"  R1@0.5:  {metrics.get('R1@0.5', 0):.4f}")
        print(f"  R1@0.7:  {metrics.get('R1@0.7', 0):.4f}")
        print(f"  Parse fail: {metrics.get('parse_fail', 0)}")
        print(f"  Time: {elapsed:.0f}s")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Comparison table ---
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    hdr = f"{'Strategy':<25} {'mIoU':>8} {'R1@0.3':>8} {'R1@0.5':>8} {'R1@0.7':>8} {'PFail':>6} {'Time':>8}"
    print(hdr)
    print("-" * 80)
    for m in all_metrics:
        print(
            f"{m.get('strategy','?'):<25} "
            f"{m.get('mIoU',0):>8.4f} "
            f"{m.get('R1@0.3',0):>8.4f} "
            f"{m.get('R1@0.5',0):>8.4f} "
            f"{m.get('R1@0.7',0):>8.4f} "
            f"{m.get('parse_fail',0):>6} "
            f"{m.get('time_sec',0):>7.0f}s"
        )
    print(f"{'='*80}")

    # --- Save comparison ---
    comp_file = os.path.join(args.output_dir, "comparison.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(comp_file, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {comp_file}")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
