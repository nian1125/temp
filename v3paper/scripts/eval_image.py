"""Image fine-grained perception evaluation.

Supports V*Bench, HR-Bench, ZoomBench.
Uses vLLM or direct model inference to evaluate image understanding.

Usage:
  # Evaluate on V*Bench
  python eval_image.py --model_path /path/to/model --benchmark vstar

  # Evaluate on HR-Bench 4K
  python eval_image.py --model_path /path/to/model --benchmark hrbench-4k

  # Evaluate from prediction file
  python eval_image.py --pred_file /path/to/predictions.jsonl --benchmark vstar
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


BENCHMARK_CONFIGS = {
    "vstar": {
        "description": "V* Bench — fine-grained visual search",
        "metric": "accuracy",
        "hf_dataset": "craigwu/vstar_bench",
    },
    "hrbench-4k": {
        "description": "HR-Bench 4K — high-resolution image understanding",
        "metric": "accuracy",
        "hf_dataset": "HaoyiDeng/HR-Bench",
        "subset": "4K",
    },
    "hrbench-8k": {
        "description": "HR-Bench 8K — ultra high-resolution",
        "metric": "accuracy",
        "hf_dataset": "HaoyiDeng/HR-Bench",
        "subset": "8K",
    },
    "zoombench": {
        "description": "ZoomBench — 6-dim fine-grained perception",
        "metric": "accuracy",
        "hf_dataset": "inclusionAI/ZoomBench",
    },
}


def extract_answer(text: str) -> str:
    """Extract answer from model response."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"(?:answer|Answer)\s*[:：]\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()

    option_match = re.search(r"\b([A-D])\b", text)
    if option_match:
        return option_match.group(1)

    return text.strip()


def compute_accuracy(predictions: list[dict]) -> dict:
    """Compute accuracy metrics."""
    correct = 0
    total = 0

    for pred in predictions:
        gt = pred.get("ground_truth", "").strip().upper()
        predicted = extract_answer(pred.get("prediction", "")).strip().upper()

        if gt and predicted:
            total += 1
            if predicted == gt or gt in predicted or predicted in gt:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def evaluate_from_predictions(pred_file: str) -> dict:
    """Evaluate from prediction JSONL file."""
    with open(pred_file) as f:
        predictions = [json.loads(line) for line in f]
    return compute_accuracy(predictions)


def evaluate_with_model(
    model_path: str,
    benchmark: str,
    max_samples: int | None = None,
    output_file: str | None = None,
) -> dict:
    """Run model inference on a benchmark dataset."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    config = BENCHMARK_CONFIGS[benchmark]

    print(f"Loading benchmark: {benchmark}")
    from datasets import load_dataset
    ds_kwargs = {"path": config["hf_dataset"]}
    if "subset" in config:
        ds_kwargs["name"] = config["subset"]
    dataset = load_dataset(**ds_kwargs, split="test")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Loading model from {model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    predictions = []
    for item in tqdm(dataset, desc=f"Evaluating {benchmark}"):
        image = item.get("image")
        question = item.get("question", item.get("problem", ""))
        gt = item.get("answer", "")

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256)

        input_len = inputs["input_ids"].shape[1]
        response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

        predictions.append({
            "question": question,
            "ground_truth": gt,
            "prediction": response,
        })

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Saved predictions to {output_file}")

    return compute_accuracy(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=list(BENCHMARK_CONFIGS.keys()))

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred_file", type=str)
    group.add_argument("--model_path", type=str)

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    if args.pred_file:
        metrics = evaluate_from_predictions(args.pred_file)
    else:
        metrics = evaluate_with_model(
            args.model_path, args.benchmark,
            args.max_samples, args.output_file,
        )

    print(f"\n{'='*50}")
    print(f"Results on {args.benchmark}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
