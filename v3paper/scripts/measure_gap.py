#!/usr/bin/env python3
"""特权信号预测器: 一次测出每个特权信号的 (Δ有用度, 边界KL可迁移性)。

每个样本跑 plain + 所有策略, 写**一条**记录(含各策略 tiou + 相对plain的边界KL),
保证 Δ 在同一批样本上对齐。KL 用 try/except 包住——即使失败, tiou/Δ 仍照常产出。

Δ      = mIoU(特权) − mIoU(plain)         越高=特权越有用 (必要条件 >0)
边界KL = KL(π(·|特权) ‖ π(·|plain)) 在GT答案的数字token上   越低=越可迁移

判据(π-Distill): 纯蒸馏要 "Δ>0 且 KL小"。numberit 应落在这。

用法(8 GPU 并行):
  for i in 0..7: CUDA_VISIBLE_DEVICES=$i python measure_gap.py \
     --model_path <base> --anno_path <jsonl> --video_dir <root> \
     --strategies plain,numberit,timestamp,red_border --max_samples 40 --chunk 8 --index $i &
输出 <output>_<index>.jsonl(每行一个样本), 之后聚合。
"""
import argparse, json, os, re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

import sys
sys.path.insert(0, "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3")
from zwz.data.tvg_opd_dataset import (
    _create_teacher_video, _read_video_frames, TEACHER_PROMPT_PREFIX,
)

GROUNDING = ("Please find the visual event described by the sentence '{}', "
             "determining its starting and ending times. The format should be: "
             "'The event happens in <start time> - <end time> seconds'.")

# 这些策略与 plain 同分辨率/同帧数, 答案token可逐位对齐 → 可算边界KL
VIDEO_COMPARABLE = {"numberit", "timestamp", "red_border", "gt_text",
                    "coarse_box", "coarse_box_hint", "numberit+red_border",
                    "coarse_text", "numberit+coarse_box", "timestamp+coarse_box"}


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    u = max(a[1], b[1]) - min(a[0], b[0])
    return max(hi - lo, 0) / u if u > 0 else 0.0


def extract_time(p):
    nums = [float(n) for n in re.findall(r"(\d+\.?\d*)", p.lower())]
    nums = nums[:len(nums) // 2 * 2]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def make_grid(video_path, rows, cols, out_path):
    frames, fps = _read_video_frames(video_path)
    if not frames:
        return None
    n = rows * cols
    idxs = np.linspace(0, len(frames) - 1, n).astype(int)
    cw, ch = 320, 180
    grid = Image.new("RGB", (cols * cw, rows * ch), (0, 0, 0))
    for j, fi in enumerate(idxs):
        fr = frames[int(fi)].convert("RGB").resize((cw, ch))
        ts = fi / fps if fps > 0 else 0.0
        d = ImageDraw.Draw(fr)
        d.text((5, 5), f"{ts:.1f}s", fill="red")
        r_, c_ = divmod(j, cols)
        grid.paste(fr, (c_ * cw, r_ * ch))
    grid.save(out_path)
    return out_path


def build_user_messages(strategy, anno, args):
    """返回 (messages, modal)。messages 只含 user 轮。"""
    vp, q = anno["video_path"], anno["query"]
    sp = anno["span"]
    gs, ge = (sp[0] if isinstance(sp[0], list) else sp)
    ds = 32
    if strategy == "grid":
        gpath = f"/tmp/grid_{os.getpid()}_{abs(hash(vp)) % 99999}.png"
        if make_grid(vp, args.grid_rows, args.grid_cols, gpath) is None:
            return None
        gtext = (f"The image is a {args.grid_rows*args.grid_cols}-frame grid uniformly sampled "
                 f"from a video (ordered left-to-right, top-to-bottom), each frame labeled with "
                 f"its timestamp in seconds. " + GROUNDING.format(q))
        return [{"role": "user", "content": [
            {"type": "image", "image": gpath},
            {"type": "text", "text": gtext}]}], "image"
    fps_use = float(args.fps)
    tot = 14336
    if strategy == "plain":
        vidpath = vp; prefix = ""
    elif strategy == "hires_frames":
        vidpath = vp; prefix = ""; fps_use = float(args.fps) * 2; tot = 14336 * 2
    elif strategy == "hires_pixels":
        vidpath = vp; prefix = ""; tot = 14336 * 2
    else:
        vidpath = _create_teacher_video(vp, float(gs), float(ge), args.cache_dir, strategy)
        prefix = TEACHER_PROMPT_PREFIX.get(strategy, "")
    content = [
        {"type": "video", "video": vidpath,
         "min_pixels": 64 * ds * ds, "total_pixels": tot * ds * ds, "fps": fps_use},
        {"type": "text", "text": prefix + GROUNDING.format(q)},
    ]
    return [{"role": "user", "content": content}], "video"


def _process(messages, modal, proc, args):
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if modal == "image":
        imgs, vids, vk = process_vision_info(messages, image_patch_size=args.image_patch_size,
                                             return_video_kwargs=True)
        return proc(text=[text], images=imgs, videos=vids, return_tensors="pt", **vk).to("cuda")
    imgs, vids, vk = process_vision_info(messages, image_patch_size=args.image_patch_size,
                                         return_video_kwargs=True, return_video_metadata=True)
    if vids is not None:
        vl, vm = zip(*vids); vl, vm = list(vl), list(vm)
    else:
        vl, vm = None, None
    return proc(text=[text], images=imgs, videos=vl, video_metadata=vm,
                return_tensors="pt", do_resize=False, **vk).to("cuda")


def predict_tiou(messages, modal, proc, model, args, gt):
    inputs = _process(messages, modal, proc, args)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    ans = proc.batch_decode([out[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0]
    ts = extract_time(ans)
    t = iou([float(gt[0]), float(gt[1])], list(ts[0])) if ts else 0.0
    return t, ans[:60]


def _find_sub(seq, sub):
    n = len(sub)
    for i in range(len(seq) - n, -1, -1):
        if seq[i:i + n] == sub:
            return i
    return -1


def _answer_logits(user_msgs, modal, answer_text, ans_ids, proc, model, args):
    """teacher-force answer_text, 返回预测各答案token的 log-prob [n_ans, V] (在cuda)。"""
    msgs = user_msgs + [{"role": "assistant",
                         "content": [{"type": "text", "text": answer_text}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if modal == "video":
        imgs, vids, vk = process_vision_info(msgs, image_patch_size=args.image_patch_size,
                                             return_video_kwargs=True, return_video_metadata=True)
        if vids is not None:
            vl, vm = zip(*vids); vl, vm = list(vl), list(vm)
        else:
            vl, vm = None, None
        inputs = proc(text=[text], images=imgs, videos=vl, video_metadata=vm,
                      return_tensors="pt", do_resize=False, **vk).to("cuda")
    else:
        imgs, vids, vk = process_vision_info(msgs, image_patch_size=args.image_patch_size,
                                             return_video_kwargs=True)
        inputs = proc(text=[text], images=imgs, videos=vids, return_tensors="pt", **vk).to("cuda")
    ids = inputs.input_ids[0].tolist()
    pos = _find_sub(ids, ans_ids)
    if pos < 0:
        return None
    with torch.no_grad():
        logits = model(**inputs).logits[0]            # [L, V]
    sel = logits[pos - 1: pos - 1 + len(ans_ids)]     # 预测每个答案token
    return F.log_softmax(sel.float(), dim=-1)


def boundary_kl(priv_msgs, plain_msgs, gt, proc, model, args):
    """KL(π(·|priv) ‖ π(·|plain)) 在GT答案数字token上的均值。失败返回 None。"""
    ans = f"The event happens in {float(gt[0]):.1f} - {float(gt[1]):.1f} seconds"
    ans_ids = proc.tokenizer(ans, add_special_tokens=False).input_ids
    lp = _answer_logits(priv_msgs, "video", ans, ans_ids, proc, model, args)
    lq = _answer_logits(plain_msgs, "video", ans, ans_ids, proc, model, args)
    if lp is None or lq is None or lp.shape != lq.shape:
        return None
    kl_tok = (lp.exp() * (lp - lq)).sum(-1)           # [n_ans]
    num_pos = [j for j, tid in enumerate(ans_ids)
               if any(c.isdigit() for c in proc.tokenizer.decode([tid]))]
    sel = kl_tok[num_pos] if num_pos else kl_tok
    return float(sel.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--anno_path", required=True)
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--strategies", default="plain,numberit")
    ap.add_argument("--output", default="/tmp/gap")
    ap.add_argument("--cache_dir", default="/tmp/gap_teacher_cache")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--image_patch_size", type=int, default=16)
    ap.add_argument("--grid_rows", type=int, default=4)
    ap.add_argument("--grid_cols", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--no_kl", action="store_true", help="跳过边界KL, 只测 Δ")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    annos = []
    for i, line in enumerate(open(args.anno_path)):
        if line.strip():
            a = json.loads(line); a["_idx"] = i
            vp = a.get("video_path", "")
            if args.video_dir and vp and not os.path.isabs(vp):
                a["video_path"] = os.path.join(args.video_dir, vp)
            annos.append(a)
    annos = annos[args.index::args.chunk]
    if args.max_samples > 0:
        annos = annos[:args.max_samples]

    il = os.path.isdir(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map="auto", trust_remote_code=True, local_files_only=il).eval()
    proc = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=il)

    strats = [s.strip() for s in args.strategies.split(",")]
    outf = open(f"{args.output}_{args.index}.jsonl", "w")
    for anno in annos:
        sp = anno["span"]; gt = (sp[0] if isinstance(sp[0], list) else sp)
        if not os.path.exists(anno["video_path"]):
            continue
        rec = {"_idx": anno["_idx"], "tiou": {}, "kl": {}}
        # plain 先跑(Δ 和 KL 都以它为基准)
        plain_built = None
        try:
            plain_built = build_user_messages("plain", anno, args)
            t, _ = predict_tiou(plain_built[0], plain_built[1], proc, model, args, gt)
            rec["tiou"]["plain"] = round(t, 4)
        except Exception as e:
            print(f"[ERR] plain idx={anno['_idx']}: {e}", flush=True)
        for s in strats:
            if s == "plain":
                continue
            try:
                built = build_user_messages(s, anno, args)
                if built is None:
                    continue
                t, _ = predict_tiou(built[0], built[1], proc, model, args, gt)
                rec["tiou"][s] = round(t, 4)
                if (not args.no_kl and s in VIDEO_COMPARABLE
                        and plain_built is not None and built[1] == "video"):
                    try:
                        kl = boundary_kl(built[0], plain_built[0], gt, proc, model, args)
                        if kl is not None:
                            rec["kl"][s] = round(kl, 4)
                    except Exception as e:
                        print(f"[KL-ERR] {s} idx={anno['_idx']}: {e}", flush=True)
            except Exception as e:
                print(f"[ERR] {s} idx={anno['_idx']}: {e}", flush=True)
        outf.write(json.dumps(rec) + "\n"); outf.flush()
    outf.close()
    print(f"[DONE] index={args.index} n={len(annos)}", flush=True)


if __name__ == "__main__":
    main()
