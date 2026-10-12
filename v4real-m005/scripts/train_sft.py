"""SFT baseline training script (for comparison only, NOT the main method).

NOTE: The main Temporal-ZwZ method uses RL (ZwZ-style or OPD-style), not SFT.
This script trains a vanilla SFT baseline for comparison in ablation tables.
Compatible with Qwen2.5-VL / Qwen3-VL models.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from qwen_vl_utils import process_vision_info

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TVGSFTDataset(Dataset):
    """SFT dataset for temporal video grounding."""

    def __init__(self, data_file: str, processor, max_seq_len: int = 4096,
                 num_frames: int = 32, fps: float = 2.0):
        with open(data_file, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.num_frames = num_frames
        self.fps = fps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = item["video_path"]
        query = item["query"]
        answer = item["answer"]

        ds = 32  # downsample_rate, 与 OPD/eval 一致
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": 64 * ds * ds,
                        "total_pixels": 14336 * ds * ds,
                        "fps": float(self.fps),
                        "max_frames": int(self.num_frames),
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Please find the visual event described by the sentence "
                            f"'{query}', determining its starting and ending times. "
                            f"The format should be: 'The event happens in "
                            f"<start time> - <end time> seconds'."
                        ),
                    },
                ],
            },
            {"role": "assistant", "content": answer},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # 关键: 带 video_metadata + video_kwargs, 否则 fps 退化成24、采样错误
        images, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=16,
            return_video_kwargs=True, return_video_metadata=True,
        )
        if videos is not None:
            videos_list, video_metadatas = zip(*videos)
            videos_list, video_metadatas = list(videos_list), list(video_metadatas)
        else:
            videos_list, video_metadatas = None, None
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos_list,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Create labels: mask user turns, only train on assistant response
        labels = input_ids.clone()
        # Find the assistant response start
        assistant_marker = self.processor.tokenizer.encode(
            "assistant\n", add_special_tokens=False
        )
        marker_len = len(assistant_marker)

        # Mask everything before the last assistant marker
        last_assistant_start = 0
        for i in range(len(input_ids) - marker_len):
            if input_ids[i : i + marker_len].tolist() == assistant_marker:
                last_assistant_start = i + marker_len

        labels[:last_assistant_start] = -100
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # 多模态张量(变长): 交给自定义 collator 处理
        for key in inputs:
            if key.startswith("pixel_values") or key.endswith("grid_thw") or key == "second_per_grid_ts":
                v = inputs[key]
                result[key] = v.squeeze(0) if (hasattr(v, "dim") and v.dim() > 2) else v
        return result


class VideoSFTCollator:
    """处理变长视频的 collator: 文本右padding, 视频张量按dim0拼接(Qwen-VL要求)。"""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)
        ids, attn, lbl = [], [], []
        for f in features:
            n = f["input_ids"].size(0)
            pad = max_len - n
            ids.append(torch.cat([f["input_ids"],
                       torch.full((pad,), self.pad_token_id, dtype=f["input_ids"].dtype)]))
            attn.append(torch.cat([f["attention_mask"],
                        torch.zeros(pad, dtype=f["attention_mask"].dtype)]))
            lbl.append(torch.cat([f["labels"],
                       torch.full((pad,), -100, dtype=f["labels"].dtype)]))
        batch = {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(lbl),
        }
        # 视频/图像张量: 沿 dim0 拼接(不能stack, 因为每个样本patch数不同)
        for key in ("pixel_values_videos", "pixel_values",
                    "video_grid_thw", "image_grid_thw", "second_per_grid_ts"):
            vals = [f[key] for f in features if key in f]
            if not vals:
                continue
            if torch.is_tensor(vals[0]):
                vals = [v if v.dim() >= 1 else v.unsqueeze(0) for v in vals]
                batch[key] = torch.cat(vals, dim=0)
            else:  # list/标量(如 second_per_grid_ts 可能是list)
                flat = []
                for v in vals:
                    flat.extend(v if isinstance(v, (list, tuple)) else [v])
                batch[key] = torch.tensor(flat)
        return batch


def prepare_sft_data(anno_path: str, video_dir: str, output_path: str,
                     durations_path: str | None = None):
    """Convert annotations to SFT JSONL format."""
    from tools.data_utils import load_charades_annos

    annos = load_charades_annos(anno_path, video_dir, durations_path=durations_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

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
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(annos)} SFT samples to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=0,
                        help="按步保存checkpoint(>0时启用steps策略, 方便挑选); 0=按epoch")
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    train_dataset = TVGSFTDataset(
        args.train_file, processor,
        max_seq_len=args.max_seq_len,
        num_frames=args.num_frames,
        fps=args.fps,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy=("steps" if args.save_steps > 0 else "epoch"),
        save_steps=(args.save_steps if args.save_steps > 0 else 500),
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=VideoSFTCollator(processor.tokenizer.pad_token_id),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
