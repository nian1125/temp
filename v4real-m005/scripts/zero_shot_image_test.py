"""Zero-shot test for Direction 2: Unified image+video framework.

Tests whether tiling + marking strategies help a base model understand
high-resolution images better.

Core idea: high-res image → split into overlapping grid (e.g., 3x3=9 tiles)
→ each tile keeps original resolution → add spatial markers → model inference.

Strategies:
  original:        Baseline — feed full image directly
  A_numberit:      Draw position number on each tile
  B_coordinate:    Add coordinate text token before each tile image
  D_number_coord:  A+B combined — number on tile + coordinate token
  E_sliding_grid:  T2SGrid-style sliding 2x2 sub-grids with spatial descriptions

Benchmarks (auto-downloaded from HuggingFace):
  - hrbench-4k: HR-Bench 4K
  - hrbench-8k: HR-Bench 8K
  - vstar:      V*Bench

Usage:
  python scripts/zero_shot_image_test.py \
      --model_path Qwen/Qwen2.5-VL-7B-Instruct \
      --benchmark hrbench-4k \
      --strategies original,A_numberit,B_coordinate,D_number_coord,E_sliding_grid \
      --grid_rows 3 --grid_cols 3 --overlap 0.15 \
      --max_samples 20 \
      --output_dir results/zero_shot_image/
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.image_tile import split_image_to_tiles, is_high_res, sliding_window_grids
from tools.frame_marker import annotate_frame_number

BENCHMARK_CONFIGS = {
    "vstar": {
        "hf_dataset": "craigwu/vstar_bench",
        "hf_split": "test",
        "local_subdir": "vstar",
        "image_repo": "vstar_images",
        "description": "V*Bench — fine-grained visual search",
    },
    "hrbench-4k": {
        "hf_dataset": "DreamMr/HR-Bench",
        "hf_config": "hrbench_version_split",
        "hf_split": "hrbench_4k",
        "local_subdir": "hrbench-4k",
        "description": "HR-Bench 4K — high-resolution image understanding",
    },
    "hrbench-8k": {
        "hf_dataset": "DreamMr/HR-Bench",
        "hf_config": "hrbench_version_split",
        "hf_split": "hrbench_8k",
        "local_subdir": "hrbench-8k",
        "description": "HR-Bench 8K — ultra high-resolution",
    },
}


def load_benchmark(benchmark_name, data_dir=None):
    """Load benchmark dataset, normalizing to list of {image, question, answer}.

    Tries local disk first (data_dir/subdir), falls back to HuggingFace download.
    """
    import base64
    import io
    from PIL import Image
    from datasets import load_from_disk, load_dataset

    config = BENCHMARK_CONFIGS[benchmark_name]
    local_subdir = config["local_subdir"]

    local_path = os.path.join(data_dir, local_subdir) if data_dir else None
    if local_path and os.path.isdir(local_path):
        print(f"Loading from local: {local_path}")
        raw = load_from_disk(local_path)
    else:
        print(f"Downloading from HuggingFace: {config['hf_dataset']}")
        kw = {"path": config["hf_dataset"]}
        if "hf_config" in config:
            kw["name"] = config["hf_config"]
        raw = load_dataset(**kw, split=config["hf_split"])

    samples = []
    image_base = None
    if "image_repo" in config and data_dir:
        image_base = os.path.join(data_dir, config["image_repo"])

    for item in raw:
        # --- Parse image ---
        img_field = item.get("image")
        pil_img = None
        if isinstance(img_field, Image.Image):
            pil_img = img_field
        elif isinstance(img_field, str):
            if img_field.startswith("/9j/") or img_field.startswith("iVBOR"):
                pil_img = Image.open(io.BytesIO(base64.b64decode(img_field)))
            elif image_base:
                img_path = os.path.join(image_base, img_field)
                if os.path.exists(img_path):
                    pil_img = Image.open(img_path).convert("RGB")

        if pil_img is None:
            continue

        # --- Parse question + answer ---
        if "A" in item and "B" in item:
            q = item["question"]
            opts = "\n".join(f"({k}) {item[k]}" for k in "ABCD" if item.get(k))
            question = f"{q}\n{opts}\nAnswer with the option's letter from the given choices directly."
            answer = item.get("answer", "")
        elif "text" in item:
            question = item["text"]
            answer = item.get("label", "")
        else:
            question = item.get("question", item.get("problem", ""))
            answer = str(item.get("answer", ""))

        samples.append({"image": pil_img, "question": question, "answer": answer.strip()})

    return samples

ALL_STRATEGIES = ["original", "A_numberit", "B_coordinate", "D_number_coord", "E_sliding_grid"]

SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    " The assistant first thinks about the reasoning process in the mind "
    "and then provides the user with the answer. The answer are enclosed "
    "within <answer> </answer> tags, respectively, i.e., "
    "reasoning process here <answer> answer here </answer>"
)

TILING_EXPLANATION = (
    "I have taken a high-resolution image ({w}x{h} pixels) and cropped it into "
    "a {rows}x{cols} grid of {n} overlapping tiles (with {overlap_pct}% overlap "
    "between adjacent tiles to avoid cutting objects at boundaries). "
    "The tiles are ordered left-to-right, top-to-bottom: "
    "tile 0 is top-left, tile {last} is bottom-right."
)

PROMPT_A = (
    "{tiling_desc}\n"
    "Each tile has its position number (0-{last}) drawn on it.\n\n"
    "{question}"
)

PROMPT_B = (
    "{tiling_desc}\n"
    "Each tile is preceded by a coordinate tag showing its [row, col] position "
    "in the grid (row 0 = top, col 0 = left).\n\n"
    "{question}"
)

PROMPT_D = (
    "{tiling_desc}\n"
    "Each tile has its position number (0-{last}) drawn on it, and is preceded by "
    "a coordinate tag [row, col] showing its grid position.\n\n"
    "{question}"
)

PROMPT_E_HEADER = (
    "I have taken a high-resolution image ({w}x{h} pixels) and cropped it into "
    "a {rows}x{cols} grid of {n} overlapping tiles ({overlap_pct}% overlap). "
    "To preserve detail, I show you {num_windows} sub-region images below. "
    "Each sub-region is a 2x2 block of adjacent tiles at full resolution. "
    "Sub-regions overlap — tiles at boundaries appear in multiple images for context."
)

PROMPT_E_FOOTER = (
    "\nBased on all the sub-region images above, answer the following question "
    "about the original high-resolution image:\n{question}"
)


def get_tile_coord(idx, cols):
    row, col = divmod(idx, cols)
    return row, col


def build_original_message(image, question):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]


def build_A_message(tiles, meta, question):
    """Strategy A: NumberIT on tiles."""
    numbered = [annotate_frame_number(t, i, font_size=40) for i, t in enumerate(tiles)]
    tiling_desc = TILING_EXPLANATION.format(**meta)
    prompt = PROMPT_A.format(tiling_desc=tiling_desc, question=question, **meta)

    content = []
    for tile in numbered:
        content.append({"type": "image", "image": tile})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}], numbered


def build_B_message(tiles, meta, question):
    """Strategy B: Coordinate tokens before each tile."""
    cols = meta["cols"]
    tiling_desc = TILING_EXPLANATION.format(**meta)
    prompt_tail = PROMPT_B.format(tiling_desc=tiling_desc, question=question, **meta)

    content = []
    for i, tile in enumerate(tiles):
        row, col = get_tile_coord(i, cols)
        content.append({"type": "text", "text": f"[row{row}, col{col}]"})
        content.append({"type": "image", "image": tile})
    content.append({"type": "text", "text": prompt_tail})
    return [{"role": "user", "content": content}], tiles


def build_D_message(tiles, meta, question):
    """Strategy D: NumberIT + Coordinate tokens."""
    cols = meta["cols"]
    numbered = [annotate_frame_number(t, i, font_size=40) for i, t in enumerate(tiles)]
    tiling_desc = TILING_EXPLANATION.format(**meta)
    prompt_tail = PROMPT_D.format(tiling_desc=tiling_desc, question=question, **meta)

    content = []
    for i, tile in enumerate(numbered):
        row, col = get_tile_coord(i, cols)
        content.append({"type": "text", "text": f"[row{row}, col{col}]"})
        content.append({"type": "image", "image": tile})
    content.append({"type": "text", "text": prompt_tail})
    return [{"role": "user", "content": content}], numbered


def build_E_message(tiles, meta, question):
    """Strategy E: T2SGrid-style sliding window sub-grids."""
    grids = sliding_window_grids(
        tiles, grid_rows=meta["rows"], grid_cols=meta["cols"], window_size=2,
    )

    header = PROMPT_E_HEADER.format(num_windows=len(grids), **meta)
    content = [{"type": "text", "text": header}]

    for i, (grid_img, wmeta) in enumerate(grids):
        rr = wmeta["row_range"]
        cr = wmeta["col_range"]
        content.append({"type": "text", "text": f"Sub-region {i+1}: rows {rr[0]}-{rr[1]}, cols {cr[0]}-{cr[1]}"})
        content.append({"type": "image", "image": grid_img})

    footer = PROMPT_E_FOOTER.format(question=question)
    content.append({"type": "text", "text": footer})
    return [{"role": "user", "content": content}], [g[0] for g in grids]


def run_inference(model, processor, messages, enable_thinking=False):
    import torch
    from qwen_vl_utils import process_vision_info

    tpl_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        tpl_kwargs["enable_thinking"] = enable_thinking
    try:
        text = processor.apply_chat_template(messages, **tpl_kwargs)
    except TypeError:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024)

    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)


def extract_answer(text):
    """Extract answer from model output (aligned with Vision-OPD normalize_model_answer)."""
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):].strip()
    if "<answer>" in text:
        start = text.find("<answer>")
        end = text.find("</answer>")
        if start != -1 and end != -1:
            return text[start + len("<answer>"):end].strip()
    if "Answer:" in text:
        return text[text.find("Answer:") + len("Answer:"):].strip()
    return text.strip()


def extract_first_option(text):
    """Extract MCQ option letter from prediction.

    Searches the tail of the response first (model usually concludes with the answer),
    then falls back to the full text. Avoids matching random uppercase letters from
    reasoning text by restricting the greedy fallback to very short responses.
    """
    if not text:
        return ""

    # For long text, try the tail first (last 150 chars)
    segments = [text[-150:], text] if len(text) > 150 else [text]

    for segment in segments:
        match = re.search(r"\(([A-Z])\)", segment)
        if match:
            return match.group(1)
        match = re.search(r"\b([A-Z])[\.\)\:]", segment)
        if match:
            return match.group(1)

    # Single letter fallback only for short direct answers (< 20 chars)
    if len(text) <= 20:
        match = re.search(r"([A-Z])", text)
        if match:
            return match.group(1)

    return ""


def extract_mcq_option(answer):
    """Extract option letter from ground truth (from Vision-OPD)."""
    if not isinstance(answer, str) or not answer:
        return ""
    text = answer.strip()
    pattern = r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))"
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    return ""


def first_letter_match(gt, pred_text):
    """Check if GT and prediction match by MCQ option letter (from Vision-OPD)."""
    gt_val = extract_mcq_option(gt)
    pred_val = extract_first_option(pred_text)
    return bool(gt_val and pred_val and gt_val == pred_val)


def compute_accuracy(results):
    correct = sum(1 for r in results if r["correct"])
    wrong_letter = sum(1 for r in results if r.get("judge_source") == "wrong_letter")
    no_letter = sum(1 for r in results if r.get("judge_source") == "no_letter")
    total = len(results)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "wrong_letter": wrong_letter,
        "no_letter": no_letter,
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot image tiling test (Direction 2)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=list(BENCHMARK_CONFIGS.keys()))
    parser.add_argument("--data_dir", type=str, default="data/benchmarks",
                        help="Local dir with pre-downloaded benchmarks (from save_to_disk)")
    parser.add_argument("--strategies", type=str,
                        default="original,A_numberit,B_coordinate,D_number_coord,E_sliding_grid")
    parser.add_argument("--grid_rows", type=int, default=3)
    parser.add_argument("--grid_cols", type=int, default=3)
    parser.add_argument("--overlap", type=float, default=0.15, help="Overlap ratio (0-0.3)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to test (0 = use all)")
    parser.add_argument("--output_dir", type=str, default="results/zero_shot_image")
    parser.add_argument("--save_samples", type=int, default=0)
    parser.add_argument("--add_thumbnail", action="store_true",
                        help="Prepend a low-res global thumbnail to tiling strategies (LLaVA-Next style)")
    parser.add_argument("--thumbnail_size", type=int, default=768,
                        help="Max dimension for global thumbnail (default: 768)")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable Qwen3-VL thinking mode (default: off for direct answers)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if predictions file exists (skip resume)")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]
    for s in strategies:
        if s not in ALL_STRATEGIES:
            parser.error(f"Unknown strategy: {s}. Choose from: {ALL_STRATEGIES}")

    config = BENCHMARK_CONFIGS[args.benchmark]
    print(f"Loading benchmark: {config['description']}")

    samples = load_benchmark(args.benchmark, data_dir=args.data_dir)
    print(f"Loaded {len(samples)} samples")

    if args.max_samples and args.max_samples > 0 and len(samples) > args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Using first {args.max_samples} samples")

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    os.makedirs(args.output_dir, exist_ok=True)
    all_metrics = []

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing: {strategy}")
        print(f"{'='*60}")

        # Resume: skip if predictions file already exists
        pred_file = os.path.join(args.output_dir, f"predictions_{strategy}.jsonl")
        if os.path.exists(pred_file) and not args.force:
            with open(pred_file) as f:
                results = [json.loads(line) for line in f]
            if len(results) >= len(samples):
                metrics = compute_accuracy(results)
                all_metrics.append({"strategy": strategy, "metrics": metrics})
                print(f"  Skipped (resume): {len(results)} predictions exist. Acc={metrics['accuracy']:.4f}")
                continue
            print(f"  Partial results found ({len(results)}/{len(samples)}), re-running...")

        results = []
        for idx, item in enumerate(tqdm(samples, desc=strategy)):
            image = item["image"]
            question = item["question"]
            gt = item["answer"]

            n = args.grid_rows * args.grid_cols
            meta = {
                "w": image.width, "h": image.height,
                "rows": args.grid_rows, "cols": args.grid_cols,
                "n": n, "last": n - 1,
                "overlap_pct": int(args.overlap * 100),
            }

            try:
                if strategy == "original":
                    messages = build_original_message(image, question)
                    vis_tiles = None
                else:
                    tiles = split_image_to_tiles(
                        image, grid_cols=args.grid_cols, grid_rows=args.grid_rows,
                        overlap_ratio=args.overlap,
                    )
                    builders = {
                        "A_numberit": build_A_message,
                        "B_coordinate": build_B_message,
                        "D_number_coord": build_D_message,
                        "E_sliding_grid": build_E_message,
                    }
                    messages, vis_tiles = builders[strategy](tiles, meta, question)

                if args.add_thumbnail and strategy != "original":
                    thumb = image.copy()
                    thumb.thumbnail((args.thumbnail_size, args.thumbnail_size), Image.LANCZOS)
                    thumb_prefix = [
                        {"type": "text", "text": "Here is a low-resolution overview of the full image for global context:"},
                        {"type": "image", "image": thumb},
                        {"type": "text", "text": "Below are high-resolution tiles showing fine details:"},
                    ]
                    messages[0]["content"] = thumb_prefix + messages[0]["content"]

                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

                response = run_inference(model, processor, messages, enable_thinking=args.enable_thinking)
            except Exception as e:
                print(f"  Error on sample {idx}: {e}")
                results.append({"question": question, "gt": gt, "pred": "", "correct": False})
                continue

            pred = extract_answer(response)
            is_correct = first_letter_match(gt, pred)
            pred_letter = extract_first_option(pred)
            gt_letter = extract_mcq_option(gt)
            if is_correct:
                judge_source = "mcq_letter"
            elif pred_letter and gt_letter:
                judge_source = "wrong_letter"
            else:
                judge_source = "no_letter"

            results.append({
                "query": question,
                "response": gt,
                "model_answer": response,
                "extracted_answer": pred,
                "judge": "Yes" if is_correct else "",
                "judge_source": judge_source,
                "correct": is_correct,
                "image_size": f"{image.width}x{image.height}",
            })

            if args.save_samples and idx < args.save_samples and vis_tiles:
                sample_dir = os.path.join(args.output_dir, "samples", f"{idx}_{strategy}")
                os.makedirs(sample_dir, exist_ok=True)
                for ti, tile in enumerate(vis_tiles):
                    tile.save(os.path.join(sample_dir, f"tile_{ti:02d}.jpg"))

        metrics = compute_accuracy(results)
        all_metrics.append({"strategy": strategy, "metrics": metrics})
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']}, wrong={metrics['wrong_letter']}, no_letter={metrics['no_letter']})")

        pred_file = os.path.join(args.output_dir, f"predictions_{strategy}.jsonl")
        with open(pred_file, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Free GPU memory between strategies
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Comparison: {args.benchmark} ({args.grid_rows}x{args.grid_cols}, {int(args.overlap*100)}% overlap)")
    print(f"{'='*70}")
    print(f"{'Strategy':<20} {'Accuracy':>10} {'Correct':>10} {'Wrong':>10} {'NoLetter':>10} {'Total':>8}")
    print("-" * 70)
    for entry in all_metrics:
        m = entry["metrics"]
        print(f"{entry['strategy']:<20} {m['accuracy']:>10.4f} {m['correct']:>10} {m.get('wrong_letter',0):>10} {m.get('no_letter',0):>10} {m['total']:>8}")
    print("=" * 70)

    summary = {
        "model": args.model_path,
        "benchmark": args.benchmark,
        "grid": f"{args.grid_rows}x{args.grid_cols}",
        "overlap": args.overlap,
        "results": all_metrics,
    }
    with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output_dir}/comparison.json")


if __name__ == "__main__":
    main()
