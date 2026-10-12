"""Teacher inference on training data for JSD data filtering (方案 A).

For each training sample, runs Teacher (model + marked video with red border
and frame numbers) to get teacher's predicted time segment, then computes
tIoU against ground truth. Saves per-sample teacher_tIoU scores.

Supports multi-GPU parallelism via --chunk/--index (same as eval_dataloader.py).

Usage:
    # Single GPU:
    python teacher_inference.py --anno_path data.jsonl --model_path /path/to/model

    # Multi-GPU (launched by run_teacher_inference.sh):
    CUDA_VISIBLE_DEVICES=0 python teacher_inference.py --chunk 8 --index 0 &
    CUDA_VISIBLE_DEVICES=1 python teacher_inference.py --chunk 8 --index 1 &
    ...
"""

import argparse
import hashlib
import json
import os
import sys
import re

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


# ── Prompts (same as tvg_opd_dataset.py) ──

GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "The format should be: 'The event happens in "
    "<start time> - <end time> seconds'."
)

TEACHER_PREFIX = (
    "Each frame in this video has its index number overlaid. "
    "Frames with a red border fall within the target event's time range. "
)


# ── Teacher video creation (from tvg_opd_dataset.py) ──

def _add_red_border(frame, border_width=8):
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    w, h = frame.size
    for i in range(border_width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline="red")
    return frame


def _annotate_frame_number(frame, number, font_size=40):
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    font = None
    for path in [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    text = str(number)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = frame.size
    draw.text((w - tw - 4, h - th - 4 - th // 3), text, font=font, fill="red")
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


def create_teacher_video(video_path, gt_start, gt_end, cache_dir, strategy="numberit+red_border"):
    _cexp = 1.5  # coarse box expansion factor
    cc = (gt_start + gt_end) / 2.0
    ch = (gt_end - gt_start) / 2.0 * _cexp

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
        in_coarse = (cc - ch) <= ts <= (cc + ch)

        if strategy == "coarse_box":
            marked.append(_add_red_border(frame) if in_coarse else frame)
        elif strategy == "numberit+coarse_box":
            f = _annotate_frame_number(frame, i)
            if in_coarse:
                f = _add_red_border(f)
            marked.append(f)
        else:  # numberit+red_border (default)
            f = _annotate_frame_number(frame, i)
            if in_gt:
                f = _add_red_border(f)
            marked.append(f)

    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = cached_path.replace(".mp4", f".tmp{os.getpid()}.mp4")
    _write_video(marked, fps, tmp_path)
    os.rename(tmp_path, cached_path)
    return cached_path


# ── IoU & time extraction (from TimeLens) ──

def iou(a, b):
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    return max(min1 - max0, 0) / (max1 - min0) if (max1 - min0) > 0 else 0.0


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


# ── Main ──

def load_annos(anno_path, video_dir=None):
    annos = []
    with open(anno_path) as f:
        for i, line in enumerate(f):
            if line.strip():
                anno = json.loads(line)
                anno["_index"] = i
                if video_dir:
                    vp = anno.get("video_path", "")
                    if vp and not os.path.isabs(vp):
                        anno["video_path"] = os.path.join(video_dir, vp)
                annos.append(anno)
    return annos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", required=True)
    parser.add_argument("--video_dir", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True, help="Output JSONL path for scores")
    parser.add_argument("--teacher_cache_dir", default="/tmp/tvg_teacher_cache")
    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--total_tokens", type=int, default=14336)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--teacher_strategy", type=str, default="numberit+red_border",
                        choices=["numberit+red_border", "coarse_box", "numberit+coarse_box"])
    args = parser.parse_args()

    downsample_rate = 32

    # Load annotations
    annos = load_annos(args.anno_path, args.video_dir)
    print(f"Total annotations: {len(annos)}")

    # Shard for multi-GPU
    annos = annos[args.index::args.chunk]
    print(f"This shard (index={args.index}, chunk={args.chunk}): {len(annos)} samples")

    # Load model
    print(f"Loading model from {args.model_path}...")
    is_local = os.path.isdir(args.model_path)
    load_kwargs = dict(
        torch_dtype=torch.float16,   # V100 原生 fp16
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

    # Run inference (incremental save)
    output_file = f"{args.output_path}_{args.index}.jsonl"
    processed_indices = set()
    if os.path.exists(output_file):
        with open(output_file) as rf:
            for line in rf:
                if line.strip():
                    processed_indices.add(json.loads(line)["_index"])
        print(f"Resuming: {len(processed_indices)} already processed, skipping them")

    out_f = open(output_file, "a")
    stats_tious = []
    n_errors = 0
    for i, anno in enumerate(annos):
        video_path = anno.get("video_path", "")
        query = anno.get("query", "")
        span = anno.get("span", [])
        duration = anno.get("duration", 0.0)

        if isinstance(span, list) and len(span) > 0 and isinstance(span[0], list):
            gt_start, gt_end = span[0]
        elif isinstance(span, list) and len(span) == 2:
            gt_start, gt_end = span
        else:
            print(f"[SKIP] Bad span format at index {anno.get('_index', i)}: {span}")
            continue

        if not os.path.exists(video_path):
            print(f"[SKIP] Video not found: {video_path}")
            continue

        if anno.get("_index", i) in processed_indices:
            continue

        try:
            # Create teacher video
            teacher_video_path = create_teacher_video(
                video_path, float(gt_start), float(gt_end), args.teacher_cache_dir,
                strategy=args.teacher_strategy,
            )

            # Build teacher messages
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": teacher_video_path,
                        "min_pixels": int(args.min_tokens * downsample_rate * downsample_rate),
                        "total_pixels": int(args.total_tokens * downsample_rate * downsample_rate),
                        "fps": float(args.fps),
                    },
                    {"type": "text", "text": TEACHER_PREFIX + GROUNDING_PROMPT.format(query)},
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

            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, do_sample=False, temperature=None,
                    top_p=None, top_k=None, max_new_tokens=512,
                )

            generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Parse and compute tIoU
            timestamps = extract_time(answer)
            if timestamps:
                pred_start, pred_end = timestamps[0]
                teacher_tiou = iou([gt_start, gt_end], [pred_start, pred_end])
            else:
                teacher_tiou = 0.0

            record = {
                "_index": anno.get("_index", i),
                "teacher_tiou": round(teacher_tiou, 4),
                "teacher_answer": answer,
                "video_path": video_path,
                "query": query,
                "span": [gt_start, gt_end],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats_tious.append(teacher_tiou)

            if (i + 1) % 50 == 0:
                out_f.flush()
                print(f"[{i+1}/{len(annos)}] tiou={teacher_tiou:.3f} answer={answer[:60]}")

        except Exception as e:
            print(f"[ERROR] index={anno.get('_index', i)}: {e}")
            record = {
                "_index": anno.get("_index", i),
                "teacher_tiou": -1.0,
                "teacher_answer": f"ERROR: {e}",
                "video_path": video_path,
                "query": query,
                "span": [gt_start, gt_end],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            n_errors += 1

    out_f.close()

    # Stats
    if stats_tious:
        tious = stats_tious
        total_new = len(tious) + n_errors
        print(f"\n=== Shard {args.index} Stats ===")
        print(f"Total: {total_new}, Valid: {len(tious)}, Errors: {n_errors}")
        if len(processed_indices) > 0:
            print(f"Resumed from: {len(processed_indices)} previously processed")
        print(f"Teacher tIoU: mean={sum(tious)/len(tious):.4f}, "
              f"median={sorted(tious)[len(tious)//2]:.4f}")
        print(f"tIoU > 0.3: {sum(1 for t in tious if t > 0.3)} ({sum(1 for t in tious if t > 0.3)/len(tious)*100:.1f}%)")
        print(f"tIoU > 0.5: {sum(1 for t in tious if t > 0.5)} ({sum(1 for t in tious if t > 0.5)/len(tious)*100:.1f}%)")
        print(f"tIoU < 0.1: {sum(1 for t in tious if t < 0.1)} ({sum(1 for t in tious if t < 0.1)/len(tious)*100:.1f}%)")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
