#!/usr/bin/env python3
"""训练数据 student 推理 — 完全复用 TimeLens 官方配置（fp16/sdpa/processor/dataloader）
与最终评测完全一致，计算出 student_iou 用于 2D 分桶。
"""

import argparse, json, os, re, hashlib, sys
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from torch.utils.data import DataLoader, Dataset
import decord

sys.path.insert(0, "/mnt/250T_ceph/zhenjunchen/code/TimeLens")
from timelens.utils import extract_time

def iou(a, b):
    max0, min0 = max(a[0], b[0]), min(a[0], b[0])
    max1, min1 = max(a[1], b[1]), min(a[1], b[1])
    return max(min1 - max0, 0) / (max1 - min0) if (max1 - min0) > 0 else 0.0

class SimpleDataset(Dataset):
    def __init__(self, annos, processor, args):
        self.annos = annos
        self.processor = processor
        self.args = args
        self.ds_rate = 32  # Qwen3-VL downsample_rate

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        video_path = anno["video_path"]
        query = anno["query"]

        # 读视频帧
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        frames = [vr[i].asnumpy() for i in range(0, len(vr), max(1, int(round(fps / self.args.fps))))]
        from PIL import Image
        frames = [Image.fromarray(f) for f in frames]

        min_px = int(self.args.min_tokens * self.ds_rate * self.ds_rate)
        total_px = int(self.args.total_tokens * self.ds_rate * self.ds_rate)

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": float(self.args.fps),
                 "min_pixels": min_px, "total_pixels": total_px},
                {"type": "text", "text": (
                    "Please find the visual event described by the sentence '{}', "
                    "determining its starting and ending times. "
                    "The format should be: 'The event happens in "
                    "<start time> - <end time> seconds'."
                ).format(query)},
            ],
        }]

        from qwen_vl_utils import process_vision_info
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 完全复用 TimeLens GroundingDataset 的 Qwen3-VL 处理路径
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        return {"inputs": inputs, "anno": anno}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno_path", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--min_tokens", type=int, default=64)
    ap.add_argument("--total_tokens", type=int, default=14336)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    # ——— 模型加载（完全复用 TimeLens eval_dataloader.py）———
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )

    # 加载标注
    annos = []
    with open(args.anno_path) as f:
        for _idx, line in enumerate(f):
            if line.strip():
                d = json.loads(line)
                if os.path.exists(d.get("video_path", "")):
                    d["_orig_idx"] = _idx  # 原始位置（和 teacher 的 _index 对齐）
                    annos.append(d)

    annos.sort(key=lambda x: x.get("duration", 0), reverse=True)
    annos = annos[args.index :: args.chunk]
    print(f"Shard {args.index}/{args.chunk}: {len(annos)} samples")

    dataset = SimpleDataset(annos, processor, args)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                        pin_memory=True, collate_fn=lambda x: x[0])

    output_file = f"{args.output_path}_{args.index}.jsonl"
    processed = set()
    if os.path.exists(output_file):
        for line in open(output_file):
            if line.strip():
                processed.add(json.loads(line).get("_orig_idx", -1))
    print(f"Resuming: {len(processed)} already done")

    fout = open(output_file, "a")
    for i, data in enumerate(loader):
        anno = data["anno"]
        oidx = anno.get("_orig_idx", i)
        if oidx in processed:
            continue
        inputs = data["inputs"].to("cuda", non_blocking=True)

        with torch.no_grad():
            output_ids = model.generate(**inputs, do_sample=False, temperature=None,
                                         top_p=None, top_k=None, max_new_tokens=512)
            gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids
                       in zip(inputs.input_ids, output_ids)]
            answer = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]

        timestamps = extract_time(answer)
        if not timestamps:
            timestamps = [[float(anno.get("duration", 30)) + 10, float(anno.get("duration", 30)) + 20]]

        gt_span = anno.get("span", [0, 0])
        if isinstance(gt_span[0], list):
            gt_span = gt_span[0]
        pred_span = timestamps[0]
        student_iou = iou(gt_span, pred_span)

        rec = {
            "_orig_idx": anno.get("_orig_idx", i),  # 用原始索引对齐 teacher
            "student_iou": round(student_iou, 4),
            "student_answer": answer,
            "video_path": anno["video_path"],
            "query": anno.get("query", ""),
            "span": [float(gt_span[0]), float(gt_span[1])],
            "pred": pred_span,
            "duration": anno.get("duration", 0),
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(annos)}] siou={student_iou:.3f}")

    fout.close()
    print(f"Done: {len(annos)} samples → {output_file}")

if __name__ == "__main__":
    main()
