"""Evaluation script for Temporal Video Grounding.

Evaluates model predictions on Charades-STA / ActivityNet / TimeLens-Bench.
Supports both local model inference and pre-computed prediction files.

Usage:
  # Evaluate from prediction file
  python eval_tvg.py \
      --pred_file /path/to/predictions.jsonl \
      --dataset charades \
      --anno_path /path/to/charades_sta_test.txt \
      --video_dir /path/to/videos/

  # Evaluate with model inference
  python eval_tvg.py \
      --model_path /path/to/checkpoint \
      --dataset charades \
      --anno_path /path/to/charades_sta_test.txt \
      --video_dir /path/to/videos/ \
      --num_frames 32 --fps 2.0
"""

import argparse
import json
import os
import sys
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


def evaluate_from_predictions(pred_file: str, annos: list[dict]) -> dict:
    """Evaluate from a prediction JSONL file."""
    with open(pred_file, "r") as f:
        predictions = [json.loads(line) for line in f]

    pred_by_key = {}
    for pred in predictions:
        key = (pred.get("video_id", ""), pred.get("query", ""))
        pred_by_key[key] = pred

    preds = []
    gts = []
    matched = 0
    for anno in annos:
        key = (anno["video_id"], anno["query"])
        if key not in pred_by_key:
            continue
        matched += 1
        pred = pred_by_key[key]

        pred_text = pred.get("prediction", pred.get("response", ""))
        parsed = parse_timestamp_output(pred_text)
        if parsed is None:
            preds.append((0.0, 0.0))
        else:
            preds.append(parsed)
        gts.append((anno["span"][0], anno["span"][1]))

    print(f"Matched {matched}/{len(annos)} annotations")
    return evaluate_predictions(preds, gts)


def evaluate_with_model(
    model_path: str,
    annos: list[dict],
    num_frames: int = 32,
    fps: float = 2.0,
    max_samples: int | None = None,
    prompt_style: str = "timelens",
    output_file: str | None = None,
) -> dict:
    """Evaluate using model inference (requires GPU + transformers)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch

    print(f"Loading model from {model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if max_samples:
        annos = annos[:max_samples]

    PROMPT_TEMPLATE = (
        "Please find the visual event described by the sentence '{query}', "
        "determining its starting and ending times. The format should be: "
        "'The event happens in <start time> - <end time> seconds'."
    )

    preds = []
    gts = []
    results = []

    for anno in tqdm(annos, desc="Evaluating"):
        if not os.path.exists(anno["video_path"]):
            continue

        query = anno["query"]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": anno["video_path"],
                        "fps": fps,
                        "max_frames": num_frames,
                    },
                    {"type": "text", "text": PROMPT_TEMPLATE.format(query=query)},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Qwen3-VL patch handling: detect patch_size from processor
        _ps = 16
        for _attr in ("image_processor", "video_processor"):
            _p = getattr(processor, _attr, None)
            _v = getattr(_p, "patch_size", None) if _p is not None else None
            if _v: _ps = int(_v); break
        images, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=_ps,
            return_video_kwargs=True, return_video_metadata=True,
        )
        # Trim odd frame counts for Qwen3-VL timestamp interleave
        if videos is not None:
            for i in range(len(videos)):
                t, m = videos[i]
                n = int(t.shape[0]) if hasattr(t, "shape") else 0
                if n > 0 and n % 2 == 1:
                    videos[i] = (t[:-1], {**m, "frames_indices": m.get("frames_indices", [])[:n-1]} if isinstance(m, dict) else m)
            videos_list = [v[0] for v in videos]
            video_metadatas = [v[1] for v in videos]
        else:
            videos_list, video_metadatas = None, None
        inputs = processor(
            text=[text], images=images, videos=videos_list,
            video_metadata=video_metadatas,
            return_tensors="pt", do_resize=False, **video_kwargs,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256)

        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

        parsed = parse_timestamp_output(response)
        if parsed is None:
            preds.append((0.0, 0.0))
        else:
            preds.append(parsed)
        gts.append((anno["span"][0], anno["span"][1]))

        iou = compute_temporal_iou(preds[-1], gts[-1])
        results.append({
            "video_id": anno["video_id"],
            "query": query,
            "gt_span": anno["span"],
            "prediction": response,
            "pred_span": list(preds[-1]),
            "iou": iou,
        })

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved predictions to {output_file}")

    return evaluate_predictions(preds, gts)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Temporal Video Grounding")
    parser.add_argument("--dataset", type=str, default="charades",
                        choices=["charades", "anet", "timelens_charades", "timelens_anet"])
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, default="")
    parser.add_argument("--durations_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred_file", type=str, help="Pre-computed predictions JSONL")
    group.add_argument("--model_path", type=str, help="Model checkpoint for inference")

    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--prompt_style", type=str, default="timelens")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    # Load annotations
    print(f"Loading {args.dataset} annotations...")
    if args.dataset == "charades":
        annos = load_charades_annos(args.anno_path, args.video_dir, args.split, args.durations_path)
    elif args.dataset == "anet":
        annos = load_anet_annos(args.anno_path, args.video_dir, args.split)
    elif args.dataset.startswith("timelens"):
        annos = load_timelens_annos(args.anno_path, args.video_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Loaded {len(annos)} annotations")

    # Evaluate
    if args.pred_file:
        metrics = evaluate_from_predictions(args.pred_file, annos)
    else:
        metrics = evaluate_with_model(
            args.model_path, annos,
            num_frames=args.num_frames,
            fps=args.fps,
            max_samples=args.max_samples,
            prompt_style=args.prompt_style,
            output_file=args.output_file,
        )

    # Print results
    print("\n" + "=" * 50)
    print(f"Results on {args.dataset} ({args.split})")
    print("=" * 50)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
