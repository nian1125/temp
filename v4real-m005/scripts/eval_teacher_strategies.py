#!/usr/bin/env python3
"""Evaluate teacher strategies on TimeLens-Bench.

Runs model inference with teacher-style privileged info (GT-marked videos)
on the TimeLens-Bench test set. Computes R1@0.3, R1@0.5, R1@0.7, mIoU.

Usage:
    # Single GPU, default strategy:
    python scripts/eval_teacher_strategies.py \
        --bench_path /path/to/charades-timelens.json \
        --video_dir /path/to/charades_videos \
        --model_path /path/to/Qwen3-VL-8B-Instruct \
        --teacher_strategy numberit+red_border

    # Multi-GPU (8 GPUs):
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i python scripts/eval_teacher_strategies.py \
            --teacher_strategy gt_text \
            --chunk 8 --index $i &
    done
    wait
    python scripts/eval_teacher_strategies.py --aggregate_dir output/teacher_eval/gt_text
"""

import argparse
import hashlib
import json
import os
import re
import sys

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


# ── Prompts ──

GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "The format should be: 'The event happens in "
    "<start time> - <end time> seconds'."
)

TEACHER_PROMPT_PREFIX = {
    "red_border": (
        "Some frames in this video are highlighted with a red border — "
        "these frames fall within the target event's time range. "
    ),
    "numberit": (
        "Each frame in this video has its index number overlaid. "
    ),
    "numberit+red_border": (
        "Each frame in this video has its index number overlaid. "
        "Frames with a red border fall within the target event's time range. "
    ),
    "gt_text": (
        "Some frames in this video are annotated with a red 'TARGET EVENT' banner "
        "and a red border — these frames fall within the target event's time range. "
    ),
    "gt_only_numberit+red_border": (
        "Only some frames in this video have annotations. "
        "Frames with a red border and a red index number indicate "
        "the target event's time range. Unmarked frames are outside the event. "
    ),
    "timemarker": (
        "The video is split into segments. "
        "Text markers 'target event begin' and 'target event end' "
        "indicate the boundaries of the target event. "
    ),
    "timemarker+numberit+red_border": (
        "The video is split into segments. "
        "Text markers 'target event begin' and 'target event end' "
        "indicate the boundaries of the target event. "
        "Frames within the event also have a red border and frame index number. "
    ),
    "timestamp": (
        "Each frame in this video has its timestamp (in seconds) shown "
        "in the top-left corner. "
    ),
    "gt_text+timestamp": (
        "Each frame in this video has its timestamp (in seconds) shown "
        "in the top-left corner. Additionally, frames within the target event's "
        "time range are annotated with a red 'TARGET EVENT' banner and a red border. "
    ),
    "coarse_box": (
        ""  # 无文本提示, 仅在 1.5× GT 窗口帧上打红框 (指方向, 不泄漏精确边界)
    ),
    "coarse_box_hint": (
        "Some frames in this video have a red border that roughly indicates "
        "the region where the target event occurs; the exact start and end may "
        "lie slightly inside this region. "
    ),
}


# ── Video helpers ──

def _add_red_border(frame, border_width=8):
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    w, h = frame.size
    for i in range(border_width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline="red")
    return frame


def _get_font(font_size):
    for path in [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _annotate_frame_number(frame, number, font_size=40):
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    font = _get_font(font_size)
    text = str(number)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = frame.size
    draw.text((w - tw - 4, h - th - 4 - th // 3), text, font=font, fill="red")
    return frame


def _annotate_text_banner(frame, text="TARGET EVENT", font_size=28):
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


def _annotate_timestamp(frame, timestamp_sec, font_size=36):
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    font = _get_font(font_size)
    text = f"{timestamp_sec:.1f}s"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 4
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0, 180))
    draw.text((pad, pad), text, font=font, fill="yellow")
    return frame


def _read_video_frames(video_path):
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    frames = [Image.fromarray(vr[i].asnumpy()) for i in range(len(vr))]
    return frames, fps


def _write_video(frames, fps, output_path):
    writer = imageio.get_writer(output_path, format="FFMPEG", fps=float(fps),
                                codec="libx264",
                                output_params=["-preset", "ultrafast", "-pix_fmt", "yuv420p", "-loglevel", "error"])
    for frame in frames:
        writer.append_data(np.asarray(frame))
    writer.close()


def create_teacher_video(video_path, gt_start, gt_end, cache_dir, strategy):
    cache_key = hashlib.md5(
        f"{video_path}_{gt_start:.2f}_{gt_end:.2f}_{strategy}".encode()
    ).hexdigest()[:16]
    cached_path = os.path.join(cache_dir, f"teacher_{cache_key}.mp4")

    if os.path.exists(cached_path):
        return cached_path

    frames, fps = _read_video_frames(video_path)
    marked = []
    for i, frame in enumerate(frames):
        ts = i / fps if fps > 0 else 0.0
        in_gt = gt_start <= ts <= gt_end

        if strategy == "numberit+red_border":
            f = _annotate_frame_number(frame, i)
            if in_gt:
                f = _add_red_border(f)
            marked.append(f)
        elif strategy == "numberit":
            marked.append(_annotate_frame_number(frame, i))
        elif strategy == "gt_text":
            if in_gt:
                f = _add_red_border(frame)
                f = _annotate_text_banner(f, "TARGET EVENT")
            else:
                f = frame
            marked.append(f)
        elif strategy == "gt_only_numberit+red_border":
            if in_gt:
                f = _annotate_frame_number(frame, i)
                f = _add_red_border(f)
            else:
                f = frame
            marked.append(f)
        elif strategy == "timestamp":
            marked.append(_annotate_timestamp(frame, ts))
        elif strategy == "gt_text+timestamp":
            f = _annotate_timestamp(frame, ts)
            if in_gt:
                f = _add_red_border(f)
                f = _annotate_text_banner(f, "TARGET EVENT")
            marked.append(f)
        else:  # red_border
            marked.append(_add_red_border(frame) if in_gt else frame)

    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = cached_path.replace(".mp4", f".tmp{os.getpid()}.mp4")
    _write_video(marked, fps, tmp_path)
    os.rename(tmp_path, cached_path)
    return cached_path


def _sample_frames_eval(video_path, target_fps, max_frames=None):
    """Read video and sample frames at target_fps."""
    frames, orig_fps = _read_video_frames(video_path)
    if orig_fps <= 0 or len(frames) == 0:
        return [], [], orig_fps
    step = max(1, round(orig_fps / target_fps))
    indices = list(range(0, len(frames), step))
    if max_frames and len(indices) > max_frames:
        sel = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
        indices = [indices[i] for i in sel]
    sampled = [frames[i] for i in indices]
    timestamps = [i / orig_fps for i in indices]
    return sampled, timestamps, orig_fps


def build_timemarker_content(video_path, gt_start, gt_end, args, visual_strategy=None):
    """Build multi-block content with text markers around GT frames."""
    downsample_rate = 32
    frames, timestamps, _ = _sample_frames_eval(video_path, float(args.fps))

    if not frames:
        return None

    before_gt, gt_frames, after_gt = [], [], []
    for frame, ts in zip(frames, timestamps):
        if ts < gt_start:
            before_gt.append(frame)
        elif ts <= gt_end:
            gt_frames.append(frame)
        else:
            after_gt.append(frame)

    if visual_strategy == "numberit+red_border":
        offset = len(before_gt)
        gt_frames = [
            _add_red_border(_annotate_frame_number(f, offset + i))
            for i, f in enumerate(gt_frames)
        ]

    total_n = len(frames)
    base_total_px = int(args.total_tokens * downsample_rate * downsample_rate)
    base_min_px = int(args.min_tokens * downsample_rate * downsample_rate)

    def _vblock(pil_list):
        n = len(pil_list)
        proportion = n / total_n if total_n > 0 else 1.0
        return {
            "type": "video",
            "video": pil_list,
            "total_pixels": max(base_min_px, int(base_total_px * proportion)),
            "min_pixels": base_min_px,
        }

    content = []
    if before_gt:
        content.append(_vblock(before_gt))
    content.append({"type": "text", "text": "target event begin"})
    if gt_frames:
        content.append(_vblock(gt_frames))
    content.append({"type": "text", "text": "target event end"})
    if after_gt:
        content.append(_vblock(after_gt))
    return content


# ── Time extraction & IoU ──

def extract_time(paragraph):
    paragraph = paragraph.lower()
    timestamps = []
    time_regex = re.compile(
        r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b"
    )
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[:len(time_matches) // 2 * 2]
    if time_matches:
        converted = []
        for t in time_matches:
            parts = t.split(":")
            if len(parts) == 3:
                h, m = map(int, parts[:2])
                s = float(parts[2])
                converted.append(h * 3600 + m * 60 + s)
            elif len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                converted.append(m * 60 + s)
        timestamps = [(converted[i], converted[i + 1]) for i in range(0, len(converted), 2)]
    if not timestamps:
        numbers = re.findall(r"(\d+\.?\d*)", paragraph)
        numbers = [float(n) for n in numbers]
        numbers = numbers[:len(numbers) // 2 * 2]
        timestamps = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
    return timestamps


def iou(a, b):
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    return max(min1 - max0, 0) / (max1 - min0) if (max1 - min0) > 0 else 0.0


# ── Annotation loading ──

def load_timelens_bench(bench_path, video_dir):
    with open(bench_path) as f:
        raw = json.load(f)
    annos = []
    for vid, info in raw.items():
        duration = info.get("duration", 0.0)
        for ts, query in zip(info.get("spans", []), info.get("queries", [])):
            video_path = os.path.join(video_dir, f"{vid}.mp4")
            annos.append({
                "video_id": vid,
                "video_path": video_path,
                "duration": duration,
                "query": query,
                "span": [float(ts[0]), float(ts[1])],
            })
    return annos


# ── Aggregation (combine multi-GPU shards) ──

def aggregate_results(output_dir):
    import glob
    import time

    pattern = os.path.join(output_dir, "shard_*.jsonl")
    shard_files = []
    for attempt in range(3):
        shard_files = sorted(glob.glob(pattern))
        if shard_files:
            break
        print(f"Attempt {attempt+1}: no shards found yet at {pattern}, waiting 5s...")
        time.sleep(5)

    if not shard_files:
        print(f"No shard files found in {output_dir}")
        print(f"  glob pattern: {pattern}")
        print(f"  dir exists: {os.path.isdir(output_dir)}")
        if os.path.isdir(output_dir):
            print(f"  dir contents: {os.listdir(output_dir)}")
        return

    all_results = []
    for f in shard_files:
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    all_results.append(json.loads(line))

    if not all_results:
        print(f"No shard files found in {output_dir}")
        return

    tious = [r["tiou"] for r in all_results]
    n = len(tious)
    miou = sum(tious) / n
    r1_03 = sum(1 for t in tious if t >= 0.3) / n * 100
    r1_05 = sum(1 for t in tious if t >= 0.5) / n * 100
    r1_07 = sum(1 for t in tious if t >= 0.7) / n * 100

    print(f"\n{'='*60}")
    print(f"Aggregated Results ({n} samples)")
    print(f"{'='*60}")
    print(f"  R1@0.3: {r1_03:.2f}")
    print(f"  R1@0.5: {r1_05:.2f}")
    print(f"  R1@0.7: {r1_07:.2f}")
    print(f"  mIoU:   {miou*100:.2f}")

    summary_path = os.path.join(output_dir, "metrics.json")
    with open(summary_path, "w") as f:
        json.dump({"n": n, "R1@0.3": round(r1_03, 2), "R1@0.5": round(r1_05, 2),
                    "R1@0.7": round(r1_07, 2), "mIoU": round(miou * 100, 2)}, f, indent=2)
    print(f"  Saved to: {summary_path}")
    return {"R1@0.3": r1_03, "R1@0.5": r1_05, "R1@0.7": r1_07, "mIoU": miou * 100}


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_path", default=None,
                        help="TimeLens-Bench JSON path")
    parser.add_argument("--video_dir", default=None,
                        help="Directory containing Charades .mp4 files")
    parser.add_argument("--model_path", default="/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct")
    parser.add_argument("--teacher_strategy", default="numberit+red_border",
                        choices=["plain"] + list(TEACHER_PROMPT_PREFIX.keys()))
    parser.add_argument("--teacher_cache_dir", default="/tmp/tvg_teacher_eval_cache")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--total_tokens", type=int, default=14336)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--aggregate_dir", default=None,
                        help="Just aggregate existing shards (no inference)")
    args = parser.parse_args()

    # Aggregate-only mode
    if args.aggregate_dir:
        aggregate_results(args.aggregate_dir)
        return

    # Auto-detect paths
    if args.bench_path is None:
        for p in [
            "/cpfs01/user/chenzhaoyu.czy/data/timelens_bench/charades-timelens.json",
            os.path.join(os.environ.get("TIMELENS_ROOT", ""), "data/TimeLens-Bench/charades-timelens.json"),
        ]:
            if os.path.exists(p):
                args.bench_path = p
                break
        if args.bench_path is None:
            print("ERROR: --bench_path not found. Provide it explicitly.")
            return

    if args.video_dir is None:
        for p in [
            "/cpfs01/user/chenzhaoyu.czy/data/timelens_bench/videos",
            os.path.join(os.environ.get("TIMELENS_ROOT", ""), "data/TimeLens-Bench/videos/charades"),
        ]:
            if os.path.isdir(p):
                args.video_dir = p
                break
        if args.video_dir is None:
            print("ERROR: --video_dir not found. Provide path to Charades videos.")
            return

    if args.output_dir is None:
        args.output_dir = f"output/teacher_eval/{args.teacher_strategy}"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.teacher_cache_dir, exist_ok=True)

    # Load annotations
    annos = load_timelens_bench(args.bench_path, args.video_dir)
    print(f"Total annotations: {len(annos)}")

    # Shard
    annos = annos[args.index::args.chunk]
    print(f"This shard (index={args.index}, chunk={args.chunk}): {len(annos)} samples")
    print(f"Strategy: {args.teacher_strategy}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    is_local = os.path.isdir(args.model_path)
    load_kwargs = dict(
        torch_dtype=torch.float16,   # V100 原生 fp16，不用 bf16 模拟
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    if is_local:
        load_kwargs["local_files_only"] = True
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, **load_kwargs
    ).eval()

    proc_kwargs = dict(
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )
    if is_local:
        proc_kwargs["local_files_only"] = True
    processor = AutoProcessor.from_pretrained(args.model_path, **proc_kwargs)

    downsample_rate = 32
    prefix = TEACHER_PROMPT_PREFIX.get(args.teacher_strategy, "")

    # Run inference
    results = []
    for i, anno in enumerate(annos):
        video_path = anno["video_path"]
        query = anno["query"]
        gt_start, gt_end = anno["span"]

        if not os.path.exists(video_path):
            print(f"[SKIP] Video not found: {video_path}")
            continue

        try:
            if args.teacher_strategy == "plain":
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_frames": args.max_frames,
                            "min_pixels": int(args.min_tokens * downsample_rate * downsample_rate),
                            "total_pixels": int(args.total_tokens * downsample_rate * downsample_rate),
                            "fps": float(args.fps),
                        },
                        {"type": "text", "text": GROUNDING_PROMPT.format(query)},
                    ],
                }]
            elif args.teacher_strategy.startswith("timemarker"):
                visual_sub = None
                if "numberit+red_border" in args.teacher_strategy:
                    visual_sub = "numberit+red_border"
                content = build_timemarker_content(
                    video_path, gt_start, gt_end, args, visual_strategy=visual_sub,
                )
                if content is None:
                    raise RuntimeError("Failed to sample frames")
                content.append({"type": "text", "text": prefix + GROUNDING_PROMPT.format(query)})
                messages = [{"role": "user", "content": content}]
            else:
                teacher_video = create_teacher_video(
                    video_path, gt_start, gt_end,
                    args.teacher_cache_dir, args.teacher_strategy
                )
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": teacher_video,
                            "min_pixels": int(args.min_tokens * downsample_rate * downsample_rate),
                            "total_pixels": int(args.total_tokens * downsample_rate * downsample_rate),
                            "fps": float(args.fps),
                        },
                        {"type": "text", "text": prefix + GROUNDING_PROMPT.format(query)},
                    ],
                }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
            )
            if videos is not None:
                videos_list, video_metadatas = zip(*videos)
                videos_list, video_metadatas = list(videos_list), list(video_metadatas)
            else:
                videos_list, video_metadatas = None, None

            inputs = processor(
                text=[text], images=images, videos=videos_list, video_metadata=video_metadatas,
                return_tensors="pt", do_resize=False, **video_kwargs,
            ).to("cuda")

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, do_sample=False, temperature=None,
                    top_p=None, top_k=None, max_new_tokens=512,
                )

            generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            timestamps = extract_time(answer)
            if timestamps:
                pred_start, pred_end = timestamps[0]
                tiou_val = iou([gt_start, gt_end], [pred_start, pred_end])
            else:
                tiou_val = 0.0

            results.append({
                "video_id": anno["video_id"],
                "query": query,
                "span": [gt_start, gt_end],
                "pred": [pred_start, pred_end] if timestamps else None,
                "answer": answer,
                "tiou": round(tiou_val, 4),
            })

            if (i + 1) % 50 == 0 or (i + 1) == len(annos):
                curr_tious = [r["tiou"] for r in results]
                curr_miou = sum(curr_tious) / len(curr_tious)
                print(f"[{i+1}/{len(annos)}] tiou={tiou_val:.3f} "
                      f"running_mIoU={curr_miou:.4f} answer={answer[:60]}")

        except Exception as e:
            print(f"[ERROR] {anno['video_id']}: {e}")
            results.append({
                "video_id": anno["video_id"],
                "query": query,
                "span": [gt_start, gt_end],
                "pred": None,
                "answer": f"ERROR: {e}",
                "tiou": 0.0,
            })

    # Save shard results
    output_file = os.path.join(args.output_dir, f"shard_{args.index}.jsonl")
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print shard stats
    tious = [r["tiou"] for r in results]
    if tious:
        n = len(tious)
        miou = sum(tious) / n
        r1_03 = sum(1 for t in tious if t >= 0.3) / n * 100
        r1_05 = sum(1 for t in tious if t >= 0.5) / n * 100
        r1_07 = sum(1 for t in tious if t >= 0.7) / n * 100
        print(f"\n=== Shard {args.index} Results ({n} samples) ===")
        print(f"  R1@0.3: {r1_03:.2f}")
        print(f"  R1@0.5: {r1_05:.2f}")
        print(f"  R1@0.7: {r1_07:.2f}")
        print(f"  mIoU:   {miou*100:.2f}")

    print(f"Results saved to {output_file}")

    # If single-GPU (chunk=1), also print final aggregate
    if args.chunk == 1:
        aggregate_results(args.output_dir)


if __name__ == "__main__":
    main()
