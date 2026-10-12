"""Zero-shot test: compare clean video vs marked frames on temporal grounding.

Tests whether temporal marking strategies help a base Qwen2.5-VL model
perform better at video temporal grounding WITHOUT any training.

Strategies are split into two groups:
  GT-free (fair):    numberit, t2sgrid, timemarker
  GT-dependent (oracle upper bound): red_border, numberit+red_border, crop+numberit

Usage:
  python scripts/zero_shot_test.py \
      --model_path Qwen/Qwen2.5-VL-7B-Instruct \
      --dataset charades \
      --anno_path data/charades/charades_sta/charades_sta_test.txt \
      --video_dir data/charades/videos/ \
      --durations_path data/charades/charades_sta/durations.json \
      --strategies clean,numberit,t2sgrid,timemarker \
      --num_frames 32 --max_samples 100 \
      --output_dir results/zero_shot/ \
      --save_samples 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.data_utils import (
    load_charades_annos,
    load_anet_annos,
    parse_timestamp_output,
    compute_temporal_iou,
    evaluate_predictions,
)
from tools.temporal_crop import load_video_frames, extract_clip_frames
from tools.frame_marker import mark_frames
from zwz.data.tvg_opd_dataset import _create_teacher_video

GT_FREE_STRATEGIES = {"numberit", "t2sgrid", "timemarker"}
GT_DEPENDENT_STRATEGIES = {"red_border", "numberit+red_border", "crop+numberit"}
ALL_STRATEGIES = GT_FREE_STRATEGIES | GT_DEPENDENT_STRATEGIES


def load_timelens_annos(anno_path: str, video_dir: str) -> list[dict]:
    """Load TimeLens-Bench annotations (supports both JSON and JSONL)."""
    with open(anno_path, "r") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "{":
            try:
                raw = json.load(f)
                if isinstance(raw, dict) and all(isinstance(v, dict) for v in raw.values()):
                    items = raw.items()
                else:
                    items = [(raw.get("video_id", ""), raw)]
            except json.JSONDecodeError:
                f.seek(0)
                items = []
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        vid = obj.get("video_id", obj.get("vid", ""))
                        items.append((vid, obj))
        else:
            items = []
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    vid = obj.get("video_id", obj.get("vid", ""))
                    items.append((vid, obj))

    annos = []
    for vid, info in items:
        duration = info.get("duration", 0.0)
        spans = info.get("spans", info.get("timestamps", []))
        queries = info.get("queries", info.get("sentences", []))
        if isinstance(spans, list) and len(spans) > 0 and not isinstance(spans[0], list):
            spans = [spans]
            queries = [queries] if isinstance(queries, str) else [queries]
        for span, query in zip(spans, queries):
            if isinstance(query, list):
                query = query[0] if query else ""
            annos.append({
                "source": "timelens_bench",
                "video_path": os.path.join(video_dir, f"{vid}.mp4"),
                "video_id": vid,
                "duration": float(duration),
                "query": query,
                "span": [float(span[0]), float(span[1])],
            })
    return annos


def load_filtered_annos(anno_path: str, video_dir: str) -> list[dict]:
    """Load filtered JSONL annotations (training data format)."""
    annos = []
    with open(anno_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            vp = obj.get("video_path", "")
            if vp and not os.path.isabs(vp):
                vp = os.path.join(video_dir, vp)
            span = obj.get("span", [0, 0])
            if isinstance(span, list) and len(span) > 0 and isinstance(span[0], (list, tuple)):
                span = span[0]
            annos.append({
                "source": obj.get("source", "filtered"),
                "video_path": vp,
                "video_id": os.path.splitext(os.path.basename(vp))[0],
                "duration": float(obj.get("duration", 0)),
                "query": obj.get("query", ""),
                "span": [float(span[0]), float(span[1])],
            })
    return annos


def load_annos(args):
    if args.dataset == "charades":
        return load_charades_annos(args.anno_path, args.video_dir, args.split, args.durations_path)
    elif args.dataset == "anet":
        return load_anet_annos(args.anno_path, args.video_dir, args.split)
    elif args.dataset.startswith("timelens"):
        return load_timelens_annos(args.anno_path, args.video_dir)
    elif args.dataset == "filtered":
        return load_filtered_annos(args.anno_path, args.video_dir)
    raise ValueError(f"Unknown dataset: {args.dataset}")


QUERY_PROMPT = (
    "Please find the visual event described by the sentence '{query}', "
    "determining its starting and ending times. The format should be: "
    "'The event happens in <answer>START - END seconds</answer>'."
)


def build_clean_message(anno, num_frames, fps):
    """Build message for clean video condition (baseline)."""
    return [{
        "role": "user",
        "content": [
            {"type": "video", "video": anno["video_path"], "fps": fps, "max_frames": num_frames},
            {"type": "text", "text": QUERY_PROMPT.format(query=anno["query"])},
        ],
    }]


TEACHER_PROMPT_PREFIX = {
    "numberit+red_border": (
        "Each frame in this video has its index number overlaid. "
        "Frames with a red border fall within the target event's time range. "
    ),
    "red_border": (
        "Frames with a red border in this video fall within the target event's time range. "
    ),
}

VIDEO_TEACHER_STRATEGIES = {"numberit+red_border", "red_border"}


def build_teacher_video_message(anno, num_frames, fps, cache_dir, strategy="numberit+red_border"):
    """Build message for GT-dependent strategies using video mode (fair comparison with clean)."""
    teacher_video_path = _create_teacher_video(
        video_path=anno["video_path"],
        gt_start=anno["span"][0],
        gt_end=anno["span"][1],
        cache_dir=cache_dir,
        strategy=strategy,
    )
    prefix = TEACHER_PROMPT_PREFIX.get(strategy, "")
    return [{
        "role": "user",
        "content": [
            {"type": "video", "video": teacher_video_path, "fps": fps, "max_frames": num_frames},
            {"type": "text", "text": prefix + QUERY_PROMPT.format(query=anno["query"])},
        ],
    }]


def build_marked_message(anno, frames, timestamps, strategy, num_frames):
    """Build message for marked-frame conditions."""
    result = mark_frames(
        frames, timestamps, strategy=strategy,
        gt_start=anno["span"][0], gt_end=anno["span"][1],
    )
    marked_frames = result["frames"]
    tokens = result["tokens"]
    duration = anno["duration"]

    content = []

    if strategy == "t2sgrid":
        content.append({"type": "image", "image": marked_frames[0]})
        ts_str = ", ".join(f"{t:.1f}s" for t in timestamps)
        content.append({"type": "text", "text": (
            f"This grid image shows {len(frames)} frames uniformly sampled "
            f"from a {duration:.1f}-second video at timestamps [{ts_str}]. "
            f"Each frame is numbered (0-{len(frames)-1}). "
            + QUERY_PROMPT.format(query=anno["query"])
        )})

    elif strategy == "timemarker":
        for i, (frame, token) in enumerate(zip(marked_frames, tokens)):
            content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": token})
        content.append({"type": "text", "text": (
            f"\nThe above are {len(frames)} frames from a {duration:.1f}-second video, "
            f"each preceded by its timestamp. "
            + QUERY_PROMPT.format(query=anno["query"])
        )})

    else:
        for frame in marked_frames:
            content.append({"type": "image", "image": frame})
        ts_str = ", ".join(f"{t:.1f}s" for t in timestamps)
        prefix = ""
        if "numberit" in strategy:
            prefix = "Each frame is numbered sequentially. "
        if "red_border" in strategy:
            prefix += "Frames with red borders are within the target segment. "
        content.append({"type": "text", "text": (
            f"These are {len(marked_frames)} frames uniformly sampled from a "
            f"{duration:.1f}-second video at timestamps [{ts_str}]. "
            + prefix
            + QUERY_PROMPT.format(query=anno["query"])
        )})

    return [{"role": "user", "content": content}], marked_frames


def run_inference(model, processor, messages):
    """Run model inference and return decoded response text."""
    import torch
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        result = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        image_inputs, video_inputs, video_kwargs = result
    except (TypeError, ValueError):
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}

    videos_list = None
    video_metadatas = None
    if video_inputs is not None:
        videos_list, video_metadatas = [], []
        for v in video_inputs:
            if isinstance(v, tuple) and len(v) == 2:
                videos_list.append(v[0])
                video_metadatas.append(v[1])
            else:
                videos_list.append(v)
                video_metadatas.append(None)
        if all(m is None for m in video_metadatas):
            video_metadatas = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=videos_list,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)


def save_sample_frames(frames, strategy, video_id, output_dir):
    """Save marked frames for visual inspection."""
    sample_dir = os.path.join(output_dir, "samples", f"{video_id}_{strategy}")
    os.makedirs(sample_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(sample_dir, f"frame_{i:03d}.jpg"))


def print_comparison_table(all_metrics):
    """Print a formatted comparison table."""
    header = f"{'Strategy':<22} {'R1@0.3':>8} {'R1@0.5':>8} {'R1@0.7':>8} {'mIoU':>8}  {'Group':<12}"
    print("\n" + "=" * 75)
    print("Zero-Shot Comparison Results")
    print("=" * 75)
    print(header)
    print("-" * 75)
    for entry in all_metrics:
        name = entry["strategy"]
        m = entry["metrics"]
        group = entry.get("group", "")
        print(f"{name:<22} {m['R1@0.3']:>8.4f} {m['R1@0.5']:>8.4f} {m['R1@0.7']:>8.4f} {m['mIoU']:>8.4f}  {group:<12}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot marking strategy comparison")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="charades",
                        choices=["charades", "anet", "timelens_charades", "timelens_anet", "filtered"])
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--durations_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--strategies", type=str, default="clean,numberit,t2sgrid,timemarker",
                        help="Comma-separated list. Use 'all' for all strategies, 'gt_free' for fair ones.")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/zero_shot")
    parser.add_argument("--mark_max_frames", type=int, default=32,
                        help="Max frames for marked strategies (image mode). Keep low to avoid OOM.")
    parser.add_argument("--teacher_cache_dir", type=str, default="results/teacher_video_cache",
                        help="Cache directory for generated teacher videos.")
    parser.add_argument("--save_samples", type=int, default=0,
                        help="Save marked frames for the first N samples (0=disable)")
    args = parser.parse_args()

    if args.strategies == "all":
        strategies = ["clean"] + sorted(ALL_STRATEGIES)
    elif args.strategies == "gt_free":
        strategies = ["clean"] + sorted(GT_FREE_STRATEGIES)
    else:
        strategies = [s.strip() for s in args.strategies.split(",")]

    for s in strategies:
        if s != "clean" and s not in ALL_STRATEGIES:
            parser.error(f"Unknown strategy: {s}")

    annos = load_annos(args)
    print(f"Loaded {len(annos)} annotations from {args.dataset}")

    if annos:
        print(f"Sample video_path[0]: {annos[0]['video_path']}")
        print(f"  exists: {os.path.exists(annos[0]['video_path'])}")
        if len(annos) > 1:
            print(f"Sample video_path[1]: {annos[1]['video_path']}")
            print(f"  exists: {os.path.exists(annos[1]['video_path'])}")

    annos = [a for a in annos if os.path.exists(a["video_path"])]
    print(f"After filtering missing videos: {len(annos)}")

    if not annos:
        print("ERROR: No valid videos found. Check video_dir and video_path format.")
        sys.exit(1)

    if args.max_samples and len(annos) > args.max_samples:
        annos = annos[:args.max_samples]
        print(f"Using first {args.max_samples} samples")

    from transformers import AutoModelForImageTextToText as _AutoVL, AutoProcessor
    import torch

    print(f"\nLoading model from {args.model_path}...")
    model = _AutoVL.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = []

    for strategy in strategies:
        group = "baseline" if strategy == "clean" else (
            "gt_free" if strategy in GT_FREE_STRATEGIES else "gt_oracle"
        )
        print(f"\n{'='*60}")
        print(f"Testing: {strategy} ({group})")
        print(f"{'='*60}")

        preds = []
        gts = []
        results = []
        frames_cache = {}

        for idx, anno in enumerate(tqdm(annos, desc=strategy)):
            vid = anno["video_id"]

            if strategy == "clean":
                messages = build_clean_message(anno, args.num_frames, args.fps)
            elif strategy in VIDEO_TEACHER_STRATEGIES:
                try:
                    messages = build_teacher_video_message(
                        anno, args.num_frames, args.fps,
                        cache_dir=args.teacher_cache_dir,
                        strategy=strategy,
                    )
                except Exception as e:
                    print(f"  Skip {vid} (teacher video): {e}")
                    preds.append((0.0, 0.0))
                    gts.append((anno["span"][0], anno["span"][1]))
                    continue
            else:
                if vid not in frames_cache:
                    try:
                        mark_nf = args.mark_max_frames
                        if strategy == "crop+numberit":
                            frames, timestamps = extract_clip_frames(
                                anno["video_path"],
                                anno["span"][0], anno["span"][1],
                                num_frames=mark_nf,
                            )
                        else:
                            frames, timestamps = load_video_frames(
                                anno["video_path"], num_frames=mark_nf
                            )
                        frames_cache[vid] = (frames, timestamps)
                    except Exception as e:
                        print(f"  Skip {vid}: {e}")
                        preds.append((0.0, 0.0))
                        gts.append((anno["span"][0], anno["span"][1]))
                        continue

                frames, timestamps = frames_cache[vid]
                messages, marked_frames = build_marked_message(
                    anno, frames, timestamps, strategy, args.mark_max_frames
                )

                if args.save_samples and idx < args.save_samples:
                    save_sample_frames(marked_frames, strategy, vid, args.output_dir)

            try:
                response = run_inference(model, processor, messages)
            except Exception as e:
                print(f"  Inference error on {vid}: {e}")
                preds.append((0.0, 0.0))
                gts.append((anno["span"][0], anno["span"][1]))
                continue

            parsed = parse_timestamp_output(response)
            pred = parsed if parsed else (0.0, 0.0)
            preds.append(pred)
            gts.append((anno["span"][0], anno["span"][1]))

            iou = compute_temporal_iou(pred, gts[-1])

            if idx < 5:
                print(f"  [{idx}] GT={anno['span']}, pred={pred}, iou={iou:.3f}")
                print(f"       response: {response[:200]}")

            results.append({
                "video_id": vid,
                "query": anno["query"],
                "gt_span": anno["span"],
                "prediction": response,
                "pred_span": list(pred),
                "iou": iou,
            })

        frames_cache.clear()

        metrics = evaluate_predictions(preds, gts)
        all_metrics.append({"strategy": strategy, "group": group, "metrics": metrics})

        print(f"  R1@0.3={metrics['R1@0.3']:.4f}  R1@0.5={metrics['R1@0.5']:.4f}  "
              f"R1@0.7={metrics['R1@0.7']:.4f}  mIoU={metrics['mIoU']:.4f}")

        pred_file = os.path.join(args.output_dir, f"predictions_{strategy}.jsonl")
        with open(pred_file, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print_comparison_table(all_metrics)

    summary = {
        "model": args.model_path,
        "dataset": args.dataset,
        "num_samples": len(annos),
        "num_frames": args.num_frames,
        "results": all_metrics,
    }
    with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output_dir}/comparison.json")


if __name__ == "__main__":
    main()
