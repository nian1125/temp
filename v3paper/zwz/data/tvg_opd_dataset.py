"""TVG-OPD Dataset: Dual-input dataset for On-Policy Distillation.

Each sample provides BOTH:
- Student input: clean video + prompt
- Teacher input: GT-marked video + same prompt (same tokenization)

Teacher videos are generated on-the-fly (red borders on GT-span frames)
and cached to disk for reuse across epochs.
"""

import copy
import hashlib
import json
import os
import tempfile
import traceback

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

# Teacher 哑火(回退成 teacher=student)可见性计数器。
# 防止 timemarker 这类策略静默失败、把训练悄悄跑成"纯GRPO无蒸馏"。
_TEACHER_FAIL_COUNT = 0
_TEACHER_OK_COUNT = 0


GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "The format should be: 'The event happens in "
    "<start time> - <end time> seconds'."
)

# THINK_MODE=1 时启用: 先基于画面内容逐步推理, 末行再给答案。
THINK_GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "First reason step by step about what happens in the video and when the "
    "described event begins and ends, based only on the visual content you observe. "
    "Then, on the last line, give the final answer in the exact format: "
    "'The event happens in <start time> - <end time> seconds'."
)

# ANCHOR_MODE=1: grounded-CoT, 逐步给出并精修中间时间估计(TAR-TVG式)。
ANCHOR_GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "Reason step by step. After each reasoning step, output your current best estimate "
    "on its own line, exactly as 'estimate: <start> - <end>'. Start from a rough estimate "
    "and progressively refine it as you examine the visual content more carefully. "
    "Finally, on the last line, give the answer exactly as: "
    "'The event happens in <start time> - <end time> seconds'."
)

# SOFTRANGE_MODE=1: 软边界, 模型自报不确定性(配 tiou_softrange reward)。
SOFTRANGE_GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "Give the answer in the exact format: 'The event happens in <start> - <end> seconds.' "
    "Then, on the next line, state how many seconds your start/end might be off, "
    "exactly as: 'uncertainty: <u> seconds'. Be calibrated: use a small u when confident, "
    "a larger u when unsure, but keep it as tight as possible while still covering the true event."
)

def _grounding_prompt():
    import os as _os
    if _os.environ.get("SOFTRANGE_MODE"):
        return SOFTRANGE_GROUNDING_PROMPT
    if _os.environ.get("ANCHOR_MODE"):
        return ANCHOR_GROUNDING_PROMPT
    if _os.environ.get("THINK_MODE"):
        return THINK_GROUNDING_PROMPT
    return GROUNDING_PROMPT

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
    "coarse_box_hint": (
        "Some frames in this video have a red border that roughly indicates "
        "the region where the target event occurs; the exact start and end may "
        "lie slightly inside this region. "
    ),
    "numberit+coarse_box": (
        "Each frame in this video has its index number overlaid. "
        "Frames with a red border roughly indicate the region where the target "
        "event occurs; the exact start and end lie within this region. "
    ),
    "timestamp+coarse_box": (
        "Each frame in this video has its timestamp (in seconds) shown in the "
        "top-left corner. Frames with a red border roughly indicate the region "
        "where the target event occurs; the exact start and end lie within it. "
    ),
    "coarse_text": (
        "Some frames in this video are marked with a 'TARGET REGION (approx)' "
        "banner and a red border, roughly indicating where the target event "
        "occurs; the precise start and end lie somewhere within this region. "
    ),
    "coarse_box_multi": (
        "Some frames in this video have a red border that roughly indicates "
        "the region where the target event occurs. Frames with a blue border "
        "indicate other events in the video that are NOT the target. "
        "Focus on the red-bordered region for the described event. "
    ),
    "coarse_box_random_multi": (
        "Some frames in this video have a red border that roughly indicates "
        "the region where the target event occurs. Frames with a blue border "
        "indicate other events in the video that are NOT the target. "
        "Focus on the red-bordered region for the described event. "
    ),
    "gt_spotlight": (
        "Frames within the target event's time range are shown in full brightness. "
        "All other frames are darkened to reduce their visual saliency. "
        "Focus on the bright (non-darkened) frames. "
    ),
    "gt_spotlight_blur": (
        "Frames within the target event's time range are shown clearly. "
        "All other frames are blurred to reduce their visual detail. "
        "Focus on the sharp (non-blurred) frames. "
    ),
    "coarse_spotlight": (
        "Some frames in this video are shown in full brightness, roughly "
        "indicating the region where the target event occurs. All other "
        "frames are darkened. The exact start and end lie within the bright region. "
    ),
    "coarse_spotlight_blur": (
        "Some frames in this video are shown clearly, roughly "
        "indicating the region where the target event occurs. All other "
        "frames are blurred. The exact start and end lie within the clear region. "
    ),
    "gt_text+timestamp": (
        "Each frame in this video has its timestamp (in seconds) shown "
        "in the top-left corner. Additionally, frames within the target event's "
        "time range are annotated with a red 'TARGET EVENT' banner and a red border. "
    ),
}


def _build_video_content(video_path, data_args):
    downsample_rate = 32
    content = {
        "type": "video",
        "video": video_path,
        "min_pixels": int(data_args.min_tokens * downsample_rate * downsample_rate),
        "total_pixels": int(data_args.total_tokens * downsample_rate * downsample_rate),
        "fps": float(data_args.fps),
    }
    if getattr(data_args, "fps_max_frames", None) is not None:
        content["max_frames"] = int(data_args.fps_max_frames)
    return content


def _build_hifps_content(video_path, data_args, hifps):
    """均匀高fps teacher(zoom的公平实现): 整段视频按 hifps 均匀采样(student用 data_args.fps)。
    单video块 + 正确fps元数据 → Qwen的 second_per_grid_ts 算得对, 不破坏timemarker、不拉伸时间轴。
    teacher帧更密(全局含事件区)= 真感知特权, 但时间坐标和student一致(都真实秒)→ 蒸馏对齐。
    显存中性: total_pixels 不放大(和student同预算), 仅放宽帧上限 → 帧更多但每帧自动降分辨率,
    显存与student基本持平(避免高fps teacher forward OOM)。"""
    downsample_rate = 32
    ratio = min(float(hifps) / max(float(data_args.fps), 1e-6), 2.0)
    content = {
        "type": "video",
        "video": video_path,
        "min_pixels": int(data_args.min_tokens * downsample_rate * downsample_rate),
        "total_pixels": int(data_args.total_tokens * downsample_rate * downsample_rate),
        "fps": float(hifps),
    }
    if getattr(data_args, "fps_max_frames", None) is not None:
        # 帧上限按比例放宽(让高fps生效); total_pixels不变 → 帧多但每帧更小, 显存不涨
        content["max_frames"] = int(int(data_args.fps_max_frames) * ratio)
    return content


def _normalize_spans(span):
    if isinstance(span, tuple):
        return [list(span)]
    if isinstance(span, list) and len(span) > 0 and isinstance(span[0], (list, tuple)):
        return [list(s) for s in span]
    if isinstance(span, list) and len(span) == 2 and isinstance(span[0], (int, float)):
        return [span]
    raise ValueError(f"Unsupported span format: {span}")


def _format_response(spans):
    return (
        "The event happens in "
        + ", ".join([f"{s:.1f} - {e:.1f} seconds" for s, e in spans])
        + "."
    )


def _process_video_inputs(messages, processor):
    """Process messages through Qwen-VL pipeline, return tokenized inputs."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    text = [text]

    # patch size 必须随模型: Qwen3-VL=16, Qwen2.5-VL=14。写死16会让Qwen2.5的
    # patches.view reshape 崩(帧按16对齐但processor按14解析)。从processor自动读。
    _ps = 16
    for _attr in ("image_processor", "video_processor"):
        _p = getattr(processor, _attr, None)
        _v = getattr(_p, "patch_size", None) if _p is not None else None
        if _v:
            _ps = int(_v); break

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=_ps,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos is not None:
        # TimeLens-7B 的自定义 processor(processing_timelens.py)有特殊契约:
        #   ① videos 须是 (tensor, metadata) 元组列表(它内部 [v[0]/v[1]] 取); 不像Qwen3-VL用 video_metadata kwarg
        #   ② 做 timestamp interleave: cur_video_tensor[1::2]=cur_video_tensor[::2] 要求【帧数偶数】
        #      且 metadata["frames_indices"] 也要同步偶数(它用 frames_indices[::2] 算时间戳)
        # 所以: 把每个视频的 tensor 和 frames_indices 一起截成偶数; 再按 processor 类型选传参格式。
        def _trim_even(tensor, meta):
            try:
                n = int(tensor.shape[0]) if hasattr(tensor, "shape") else None
                if n is not None and n % 2 == 1:
                    tensor = tensor[:-1]
                    if isinstance(meta, dict) and "frames_indices" in meta:
                        fi = meta["frames_indices"]
                        if len(fi) == n:
                            meta = {**meta, "frames_indices": fi[:-1]}
            except Exception:
                pass
            return tensor, meta
        trimmed = [_trim_even(v[0], v[1]) for v in videos]
        videos_list = [t for t, _ in trimmed]
        video_metadatas = [m for _, m in trimmed]
        _is_timelens = "timelens" in (type(processor).__name__ + type(processor).__module__).lower()
    else:
        videos_list, video_metadatas, _is_timelens = None, None, False

    if _is_timelens and videos_list is not None:
        # TimeLens processor: videos 传 (tensor, metadata) 元组列表, 不传 video_metadata kwarg
        inputs = processor(
            text=text,
            images=images,
            videos=list(zip(videos_list, video_metadatas)),
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
    else:
        inputs = processor(
            text=text,
            images=images,
            videos=videos_list,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
    return inputs, text[0]


# ---------------------------------------------------------------------------
# Teacher video generation
# ---------------------------------------------------------------------------

def _add_red_border(frame: Image.Image, border_width: int = 8) -> Image.Image:
    """Draw a red border around a PIL frame."""
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    w, h = frame.size
    for i in range(border_width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline="red")
    return frame


def _add_blue_border(frame: Image.Image, border_width: int = 8) -> Image.Image:
    """Draw a blue border around a PIL frame (for non-target events)."""
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    w, h = frame.size
    for i in range(border_width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline="blue")
    return frame


def _darken_frame(frame: Image.Image, factor: float = 0.35) -> Image.Image:
    """Darken a frame by multiplying brightness (0=black, 1=original)."""
    from PIL import ImageEnhance
    return ImageEnhance.Brightness(frame.copy()).enhance(factor)


def _blur_frame(frame: Image.Image, radius: float = 8.0) -> Image.Image:
    """Gaussian blur a frame."""
    from PIL import ImageFilter
    return frame.copy().filter(ImageFilter.GaussianBlur(radius=radius))
    w, h = frame.size
    for i in range(border_width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline="blue")
    return frame


def _annotate_frame_number(frame: Image.Image, number: int, font_size: int = 40) -> Image.Image:
    """Draw frame index number on bottom-right corner."""
    from PIL import ImageFont
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


def _annotate_text_banner(frame: Image.Image, text: str = "TARGET EVENT",
                          font_size: int = 28, bg_color="red", text_color="white") -> Image.Image:
    """Draw a text banner at the top of a PIL frame."""
    from PIL import ImageFont
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
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, _ = frame.size
    pad = 4
    bar_h = th + 2 * pad
    draw.rectangle([0, 0, w, bar_h], fill=bg_color)
    draw.text(((w - tw) // 2, pad), text, font=font, fill=text_color)
    return frame


def _annotate_timestamp(frame: Image.Image, timestamp_sec: float,
                        font_size: int = 36) -> Image.Image:
    """Draw timestamp (e.g. '5.2s') in the top-left corner of a PIL frame."""
    from PIL import ImageFont
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
    text = f"{timestamp_sec:.1f}s"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 4
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0, 180))
    draw.text((pad, pad), text, font=font, fill="yellow")
    return frame


def _read_video_frames(video_path: str):
    """Read all frames from video using decord. Returns (frames, fps)."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    frames = [Image.fromarray(vr[i].asnumpy()) for i in range(len(vr))]
    return frames, fps


def _write_video(frames, fps, output_path):
    """Write PIL frames to mp4 via imageio (bundles its own ffmpeg).

    macro_block_size=1 禁止 imageio 自动把尺寸 padding 到 16 倍数(否则 360→368,
    与读回时 do_resize=False 的 patch 网格不一致 → reshape 报错)。再把每帧裁成
    偶数宽高以满足 yuv420p 要求。
    """
    def _to_even(arr):
        h, w = arr.shape[:2]
        return arr[: h - (h % 2), : w - (w % 2)]

    writer = imageio.get_writer(output_path, format="FFMPEG", fps=float(fps), codec="libx264",
                                macro_block_size=1,
                                output_params=["-preset", "ultrafast", "-pix_fmt", "yuv420p"])
    for frame in frames:
        writer.append_data(_to_even(np.asarray(frame)))
    writer.close()


def _create_teacher_video(video_path, gt_start, gt_end, cache_dir, strategy="numberit+red_border",
                          other_spans=None, zoom_expand=1.5, blue_zoom_expand=None,
                          video_duration=None):
    """Create GT-marked video for teacher.

    Strategies:
    - red_border: red border on GT-span frames only
    - numberit+red_border: frame index on all frames + red border on GT frames
    - gt_text: 'TARGET EVENT' banner + red border on GT frames
    - gt_only_numberit+red_border: frame index + red border on GT frames only
    - coarse_box_multi: GT=red border, other events=blue border (coarse 1.5x window)

    other_spans: list of [start, end] for other events in the same video.

    blue_zoom_expand: 蓝框(干扰事件)窗口的扩张系数。
        None  → 沿用 zoom_expand (历史行为, 蓝框也被扩 1.5×)
        1.0   → 论文口径: 干扰窗口保留其 verified 边界, 不扩张。
                蓝框扩 1.5× 会向两侧溢出、可能盖到红框附近, 使"哪里不是目标"这条
                干扰信息本身变得不可靠, 所以论文只扩红框。
    video_duration: 已知视频时长, 用于把窗口 clip 到 [0, duration]; None 则用解码帧数推断。
    Caches to disk for reuse.
    """
    _bexp = zoom_expand if blue_zoom_expand is None else float(blue_zoom_expand)
    cache_key = hashlib.md5(
        f"{video_path}_{gt_start:.2f}_{gt_end:.2f}_{strategy}_{other_spans}"
        f"_ze{zoom_expand:.2f}_bz{_bexp:.2f}".encode()
    ).hexdigest()[:16]
    cached_path = os.path.join(cache_dir, f"teacher_{cache_key}.mp4")

    if os.path.exists(cached_path):
        return cached_path

    frames, fps = _read_video_frames(video_path)
    marked = []
    _dur = float(video_duration) if video_duration else (len(frames) / fps if fps > 0 else 0.0)

    def _clip(lo, hi):
        """把窗口夹到 [0, duration], 防越界(原代码未 clip)。"""
        if _dur > 0:
            lo, hi = max(0.0, lo), min(_dur, hi)
        else:
            lo = max(0.0, lo)
        return lo, hi

    # coarse 放大窗口 (所有 coarse_* 策略共用: 指方向, 不泄漏精确边界)
    _cexp = zoom_expand  # 从参数传入, 不再硬编码 1.5
    _cc = (gt_start + gt_end) / 2.0
    _ch = (gt_end - gt_start) / 2.0 * _cexp

    # Precompute coarse windows for other events (for coarse_box_multi)
    _other_coarse_windows = []
    if other_spans and strategy == "coarse_box_multi":
        for o_start, o_end in other_spans:
            o_cc = (o_start + o_end) / 2.0
            o_ch = (o_end - o_start) / 2.0 * _bexp   # 蓝框用独立系数(论文: 1.0=不扩)
            _other_coarse_windows.append(_clip(o_cc - o_ch, o_cc + o_ch))

    # Precompute random blue windows (for coarse_box_random_multi)
    _random_coarse_windows = []
    if strategy == "coarse_box_random_multi":
        import random as _rng
        _seed_str = f"{video_path}_{gt_start:.2f}_{gt_end:.2f}"
        _rng.seed(int(hashlib.md5(_seed_str.encode()).hexdigest()[:8], 16))
        video_dur = len(frames) / fps if fps > 0 else 0
        n_blue = len(other_spans) if other_spans else 1
        # 参考窗口大小: 用 other_spans 平均时长 或 GT 时长
        if other_spans:
            avg_dur = sum(e - s for s, e in other_spans) / len(other_spans)
        else:
            avg_dur = gt_end - gt_start
        avg_dur = max(avg_dur, 2.0)
        gt_lo, gt_hi = _cc - _ch, _cc + _ch
        for _ in range(n_blue):
            for _attempt in range(20):
                win_size = avg_dur * _cexp
                max_start = max(0.0, video_dur - win_size)
                if max_start <= 0:
                    break
                rs = _rng.uniform(0, max_start)
                re_ = rs + win_size
                # 不与 GT coarse window 重叠
                if re_ < gt_lo or rs > gt_hi:
                    _random_coarse_windows.append((rs, re_))
                    break

    _gt_lo_c, _gt_hi_c = _clip(_cc - _ch, _cc + _ch)   # 红框窗口也 clip 到 [0, duration]
    for i, frame in enumerate(frames):
        ts = i / fps if fps > 0 else 0.0
        in_gt = gt_start <= ts <= gt_end
        in_coarse = _gt_lo_c <= ts <= _gt_hi_c

        if strategy == "coarse_box_multi":
            # GT red (highest priority), other events blue
            if in_coarse:
                marked.append(_add_red_border(frame))
            elif any(lo <= ts <= hi for lo, hi in _other_coarse_windows):
                marked.append(_add_blue_border(frame))
            else:
                marked.append(frame)
        elif strategy == "coarse_box_random_multi":
            # GT red, random blue boxes (验证蓝框语义作用)
            if in_coarse:
                marked.append(_add_red_border(frame))
            elif any(lo <= ts <= hi for lo, hi in _random_coarse_windows):
                marked.append(_add_blue_border(frame))
            else:
                marked.append(frame)
        elif strategy == "gt_spotlight":
            # GT frames normal, non-GT frames darkened
            marked.append(frame if in_gt else _darken_frame(frame))
        elif strategy == "gt_spotlight_blur":
            # GT frames normal, non-GT frames blurred
            marked.append(frame if in_gt else _blur_frame(frame))
        elif strategy == "coarse_spotlight":
            # Coarse window frames normal, others darkened (non-leaking)
            marked.append(frame if in_coarse else _darken_frame(frame))
        elif strategy == "coarse_spotlight_blur":
            # Coarse window frames normal, others blurred (non-leaking)
            marked.append(frame if in_coarse else _blur_frame(frame))
        elif strategy == "numberit+red_border":
            f = _annotate_frame_number(frame, i)
            if in_gt:
                f = _add_red_border(f)
            marked.append(f)
        elif strategy == "numberit":
            marked.append(_annotate_frame_number(frame, i))
        elif strategy in ("coarse_box", "coarse_box_hint"):
            # ZwZ式粗时间框: 在GT放大窗口内的帧加红框(指方向, 不泄漏精确边界)
            # coarse_box: 无prompt解释; coarse_box_hint: 配温和prompt(见TEACHER_PROMPT_PREFIX)
            marked.append(_add_red_border(frame) if in_coarse else frame)
        elif strategy == "numberit+coarse_box":
            # v8c(numberit+red_border)的非泄漏版: 帧号 + 粗窗口红框(不标精确边界)
            f = _annotate_frame_number(frame, i)
            if in_coarse:
                f = _add_red_border(f)
            marked.append(f)
        elif strategy == "timestamp+coarse_box":
            # 时间戳 + 粗窗口红框(绝对时间参照 + 大致区域, 不给精确答案)
            f = _annotate_timestamp(frame, ts)
            if in_coarse:
                f = _add_red_border(f)
            marked.append(f)
        elif strategy == "coarse_text":
            # gt_text的非泄漏版: 在粗窗口帧上打"approx"banner+红框(不是精确帧, 不标精确边界)
            if in_coarse:
                f = _add_red_border(frame)
                f = _annotate_text_banner(f, "TARGET REGION (approx)")
            else:
                f = frame
            marked.append(f)
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
        else:
            marked.append(_add_red_border(frame) if in_gt else frame)

    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = cached_path.replace(".mp4", f".tmp{os.getpid()}.mp4")
    _write_video(marked, fps, tmp_path)
    os.rename(tmp_path, cached_path)
    return cached_path


def _sample_frames(video_path, target_fps, max_frames=None):
    """Read video and sample frames at target_fps. Returns (frames, timestamps, orig_fps)."""
    frames, orig_fps = _read_video_frames(video_path)
    if orig_fps <= 0 or len(frames) == 0:
        return [], [], orig_fps

    total_dur = len(frames) / orig_fps
    step = max(1, round(orig_fps / target_fps))
    indices = list(range(0, len(frames), step))
    if max_frames and len(indices) > max_frames:
        sel = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
        indices = [indices[i] for i in sel]

    sampled = [frames[i] for i in indices]
    timestamps = [i / orig_fps for i in indices]
    return sampled, timestamps, orig_fps


def _build_timemarker_content(video_path, gt_start, gt_end, data_args, visual_strategy=None):
    """Build teacher content with text markers around GT frame segments.

    Splits sampled frames into 3 groups (before_gt, gt, after_gt) and inserts
    'target event begin' / 'target event end' text tokens between them.

    Args:
        visual_strategy: None for pure timemarker; "numberit+red_border" for combined.
    Returns:
        list of content blocks (without the final prompt text block).
    """
    downsample_rate = 32
    target_fps = float(data_args.fps)
    max_frames = int(data_args.fps_max_frames) if getattr(data_args, "fps_max_frames", None) else None

    frames, timestamps, _ = _sample_frames(video_path, target_fps, max_frames)
    if not frames:
        return [_build_video_content(video_path, data_args)]

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
    base_total_px = int(data_args.total_tokens * downsample_rate * downsample_rate)
    base_min_px = int(data_args.min_tokens * downsample_rate * downsample_rate)

    def _video_block(pil_list):
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
        content.append(_video_block(before_gt))
    content.append({"type": "text", "text": "target event begin"})
    if gt_frames:
        content.append(_video_block(gt_frames))
    content.append({"type": "text", "text": "target event end"})
    if after_gt:
        content.append(_video_block(after_gt))

    return content


def _make_grid_image(frames, tss, rows, cols, cell_w=224, cell_h=128):
    """把一组帧拼成 rows×cols 的复合grid图(行优先), 每格左上角标该帧时间戳。
    格子尺寸必须32对齐(Qwen3-VL patch16×merge2; do_resize=False下不对齐会reshape崩)。
    """
    from PIL import ImageFont
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for j, (f, t) in enumerate(zip(frames, tss)):
        if j >= rows * cols:
            break
        cell = f.convert("RGB").resize((cell_w, cell_h))
        d = ImageDraw.Draw(cell)
        lab = f"{t:.1f}s"
        d.rectangle([0, 0, 11 * len(lab), 20], fill=(0, 0, 0))
        d.text((2, 2), lab, font=font, fill="yellow")
        r_, c_ = divmod(j, cols)
        grid.paste(cell, (c_ * cell_w, r_ * cell_h))
    return grid


def _build_t2sgrid_content(video_path, data_args, rows=3, cols=3, overlap=1):
    """T2SGrid式表示: 重叠滑窗切clip, 每窗帧按时序拼成一张grid图(行优先)+ 复合文本时间戳。
    多张grid序列(非单张大montage)。不依赖GT(纯表示, train/test通用)。
    """
    base_fps = float(data_args.fps)
    max_frames = int(data_args.fps_max_frames) if getattr(data_args, "fps_max_frames", None) else None
    rows = int(getattr(data_args, "grid_rows", rows))
    cols = int(getattr(data_args, "grid_cols", cols))
    overlap = int(getattr(data_args, "grid_overlap", overlap))

    frames, ts, _ = _sample_frames(video_path, base_fps, max_frames)
    if not frames:
        return [_build_video_content(video_path, data_args)]
    win = rows * cols
    stride = max(1, win - overlap)
    content = [{"type": "text",
                "text": ("This video is shown as a sequence of grid images. Each grid packs "
                         f"{win} consecutive frames in row-major (left-to-right, top-to-bottom) "
                         "chronological order; each cell shows that frame's timestamp in seconds.")}]
    for s in range(0, len(frames), stride):
        wf = frames[s:s + win]; wt = ts[s:s + win]
        if not wf:
            break
        gimg = _make_grid_image(wf, wt, rows, cols)
        content.append({"type": "text", "text": f"Clip [{wt[0]:.1f}-{wt[-1]:.1f}s]:"})
        content.append({"type": "image", "image": gimg})
        if s + win >= len(frames):
            break
    return content


def _build_zoom_content(video_path, gt_start, gt_end, data_args, zoom_mult=3.0):
    """时间zoom特权: 在【GT附近放大的窗口】(不是精确GT)用 zoom_mult 倍fps密采,
    上下文稀疏。窗口比GT宽(zoom_expand倍, 默认2×; 至少±zoom_min_half秒),
    使事件边界落在密集区【内部】→ teacher须自己在窗口内定位(降软泄漏, 逼出忠实推理)。
    marker中性(只说"此段更高帧率", 不声称这是事件区)。复用帧级框架, 不裁剪不丢绝对时间。
    zoom_expand=1.0 → 退回精确GT(消融用)。
    """
    downsample_rate = 32
    base_fps = float(data_args.fps)
    max_frames = int(data_args.fps_max_frames) if getattr(data_args, "fps_max_frames", None) else None
    zoom_expand = float(getattr(data_args, "zoom_expand", 2.0))    # 窗口=GT宽×expand; 1.0=精确GT
    zoom_min_half = float(getattr(data_args, "zoom_min_half", 2.0))  # 窗口至少±2秒(短事件别太窄)

    # GT附近的放大窗口(比GT宽, 边界落在内部)
    _c = (gt_start + gt_end) / 2.0
    _half = max((gt_end - gt_start) / 2.0 * zoom_expand, zoom_min_half)
    w_start = max(0.0, _c - _half)
    w_end = _c + _half

    # 稀疏上下文(全片base fps)
    frames, ts, _ = _sample_frames(video_path, base_fps, max_frames)
    if not frames:
        return [_build_video_content(video_path, data_args)]
    before = [f for f, t in zip(frames, ts) if t < w_start]
    after = [f for f, t in zip(frames, ts) if t > w_end]
    # 放大窗口高fps密采(zoom)
    dframes, dts, _ = _sample_frames(video_path, base_fps * zoom_mult, max_frames)
    win_dense = [f for f, t in zip(dframes, dts) if w_start <= t <= w_end]
    if not win_dense:  # 窗口太短, 退回稀疏
        win_dense = [f for f, t in zip(frames, ts) if w_start <= t <= w_end]

    total_n = len(before) + len(win_dense) + len(after)
    base_total_px = int(data_args.total_tokens * downsample_rate * downsample_rate)
    base_min_px = int(data_args.min_tokens * downsample_rate * downsample_rate)

    def _vb(pil_list):
        n = len(pil_list)
        prop = n / total_n if total_n > 0 else 1.0
        return {"type": "video", "video": pil_list,
                "total_pixels": max(base_min_px, int(base_total_px * prop)),
                "min_pixels": base_min_px}

    content = []
    if before:
        content.append(_vb(before))
    content.append({"type": "text", "text": "[the following segment is shown at a higher frame rate]"})
    content.append(_vb(win_dense))
    content.append({"type": "text", "text": "[end of higher-frame-rate segment]"})
    if after:
        content.append(_vb(after))
    return content


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def load_timelens_annos(anno_path, video_dir):
    """Load TimeLens-Bench format annotations."""
    with open(anno_path) as f:
        raw = json.load(f)

    annos = []
    for vid, info in raw.items():
        duration = info.get("duration", 0.0)
        for ts, query in zip(info.get("spans", []), info.get("queries", [])):
            video_path = os.path.join(video_dir, f"{vid}.mp4")
            annos.append({
                "source": "timelens",
                "video_path": video_path,
                "video_id": vid,
                "duration": duration,
                "query": query,
                "span": [float(ts[0]), float(ts[1])],
            })
    return annos


def load_filtered_annos(anno_path):
    """Load pre-filtered annotations (JSONL or JSON list)."""
    if anno_path.endswith(".jsonl"):
        annos = []
        with open(anno_path) as f:
            for line in f:
                if line.strip():
                    annos.append(json.loads(line))
        return annos
    with open(anno_path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TVGOPDDataset(Dataset):
    """Dataset for TVG On-Policy Distillation training.

    Supports two modes:
    - sft: Supervised fine-tuning with teacher silver labels
    - grpo: GRPO training with dual (student + teacher) inputs
    """

    def __init__(
        self,
        processor,
        data_args,
        training_mode="grpo",
        teacher_cache_dir=None,
        teacher_strategy="red_border",
        student_mark_strategy=None,
        ref_anno_path=None,
        teacher_prompt_override=None,
    ):
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        self.training_mode = training_mode
        self.teacher_strategy = teacher_strategy
        self.student_mark_strategy = student_mark_strategy  # ZwZ式: 给student输入也加标记(B/B1)
        self.teacher_cache_dir = teacher_cache_dir or tempfile.mkdtemp(prefix="tvg_teacher_")
        self.teacher_prompt_override = teacher_prompt_override  # 自定义teacher prompt模板(含{query})
        # zoom_expand: 从 data_args 读取, 传入 _create_teacher_video 控制粗窗口大小
        self.zoom_expand = float(getattr(data_args, "zoom_expand", 1.5))
        # blue_zoom_expand: 蓝框独立扩张系数。<0 或缺省 → None → 沿用 zoom_expand(历史行为)。
        # 论文口径设 1.0: 干扰窗口保留 verified 边界, 只有红框扩 1.5×。
        _bz = float(getattr(data_args, "blue_zoom_expand", -1.0))
        self.blue_zoom_expand = None if _bz < 0 else _bz

        raw_annos = self._load_annos()
        self.annos = self._filter_annos(raw_annos)
        # Build video → all_spans mapping for multi-event teacher strategies
        # ref_anno_path: 用参考数据集(如 TimeLens-100K)构建蓝框映射, 与训练数据分开
        ref_annos = self._load_ref_annos(ref_anno_path) if ref_anno_path else self.annos
        self._video_all_spans = {}
        for anno in ref_annos:
            vp = anno["video_path"]
            for span_pair in anno["span"]:
                s, e = float(span_pair[0]), float(span_pair[1])
                self._video_all_spans.setdefault(vp, set()).add((round(s, 2), round(e, 2)))
        multi_count = sum(1 for v in self._video_all_spans.values() if len(v) > 1)
        # 蓝框覆盖率: coarse_box_multi 的干扰信息只有 >1 事件的视频才有。
        # 用训练集自身当 ref 时该比例受采样偏差影响, 论文口径应 --ref_anno_path 指全量标注。
        _blue_ok = sum(1 for a in self.annos
                       if len(self._video_all_spans.get(a["video_path"], ())) > 1)
        print(f"TVGOPDDataset: {len(self.annos)} samples "
              f"(from {len(raw_annos)} raw), mode={training_mode}, "
              f"ref_anno_path={'custom' if ref_anno_path else 'self'}, "
              f"ref_samples={len(ref_annos)}, "
              f"videos_with_multi_events={multi_count}, "
              f"samples_with_blue_box={_blue_ok}/{len(self.annos)} "
              f"({100.0 * _blue_ok / max(len(self.annos), 1):.1f}%), "
              f"blue_zoom_expand={self.blue_zoom_expand if self.blue_zoom_expand is not None else 'same_as_red'}")

    def _load_ref_annos(self, ref_anno_path):
        """Load reference annotation file for building _video_all_spans (blue-box multi-event mapping).

        Returns list of dicts with keys: video_path, span.
        Each span is [[start, end]].
        """
        ref_annos = []
        with open(ref_anno_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                vp = d.get("video_path", "")
                if not vp:
                    continue
                # 补 video_dir 前缀 (与 _load_annos 一致)
                video_dir = getattr(self.data_args, "video_dir", None)
                if video_dir and not os.path.isabs(vp):
                    vp = os.path.join(video_dir, vp)
                span = d.get("span", [])
                if not span:
                    continue
                # span 可能是 [[s,e]] 或 [s,e]
                if isinstance(span[0], (int, float)):
                    span = [span]
                for sp in span:
                    ref_annos.append({"video_path": vp, "span": [[float(sp[0]), float(sp[1])]]})
        return ref_annos

    def _load_annos(self):
        anno_path = self.data_args.anno_path
        video_dir = getattr(self.data_args, "video_dir", None)
        dataset = getattr(self.data_args, "dataset_name", "timelens")

        if dataset.startswith("timelens") and video_dir:
            return load_timelens_annos(anno_path, video_dir)

        annos = load_filtered_annos(anno_path)
        if video_dir:
            for anno in annos:
                vp = anno.get("video_path", "")
                if vp and not os.path.isabs(vp):
                    anno["video_path"] = os.path.join(video_dir, vp)
        return annos

    def _filter_annos(self, annos):
        filtered = []
        skip_duration = 0
        skip_span = 0
        skip_path = 0
        shown = 0
        for anno in annos:
            duration = anno.get("duration", float("inf"))
            min_len = getattr(self.data_args, "min_video_len", -1)
            max_len = getattr(self.data_args, "max_video_len", -1)
            if min_len >= 0 and duration < min_len:
                skip_duration += 1
                continue
            if max_len >= 0 and duration > max_len:
                skip_duration += 1
                continue
            anno = dict(anno)
            try:
                anno["span"] = _normalize_spans(anno["span"])
            except (ValueError, KeyError):
                skip_span += 1
                continue
            vp = anno.get("video_path", "")
            if not os.path.exists(vp):
                skip_path += 1
                if shown < 3:
                    print(f"  [filter] video not found: {vp}")
                    shown += 1
                continue
            filtered.append(anno)
        if shown < 3 and annos:
            a0 = annos[0]
            print(f"  [filter] sample keys: {list(a0.keys())}")
            print(f"  [filter] sample video_path: {a0.get('video_path', 'N/A')}")
        print(f"  [filter] skip_duration={skip_duration}, skip_span={skip_span}, "
              f"skip_path={skip_path}, kept={len(filtered)}")
        return filtered

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        if self.training_mode == "sft":
            return self._getitem_sft(idx)
        return self._getitem_grpo(idx)

    def _getitem_sft(self, idx):
        """SFT mode: clean video + ground truth response."""
        anno = copy.deepcopy(self.annos[idx])
        spans = anno["span"]

        messages = [
            {
                "role": "user",
                "content": [
                    _build_video_content(anno["video_path"], self.data_args),
                    {"type": "text", "text": GROUNDING_PROMPT.format(anno["query"])},
                ],
            }
        ]

        response = _format_response(spans)
        messages.append({"role": "assistant", "content": response})

        inputs, _ = _process_video_inputs(messages, self.processor)
        inputs["input_ids"] = inputs["input_ids"][0]
        # Qwen3-VL: mm_token_type_ids 是逐token序列, 与 input_ids 同步去掉 batch 维
        if "mm_token_type_ids" in inputs and inputs["mm_token_type_ids"].dim() > 1:
            inputs["mm_token_type_ids"] = inputs["mm_token_type_ids"][0]
        return inputs

    def _getitem_grpo(self, idx):
        """GRPO mode: returns student + teacher video inputs."""
        global _TEACHER_FAIL_COUNT, _TEACHER_OK_COUNT
        anno = copy.deepcopy(self.annos[idx])
        gt_start, gt_end = anno["span"][0]

        student_prompt_text = {
            "type": "text",
            "text": _grounding_prompt().format(anno["query"]),
        }

        # Teacher prompt: 优先用 teacher_prompt_override, 否则用策略默认
        if self.teacher_prompt_override:
            teacher_prompt_text = {
                "type": "text",
                "text": self.teacher_prompt_override.format(anno["query"]),
            }
        else:
            prefix = TEACHER_PROMPT_PREFIX.get(self.teacher_strategy, "")
            teacher_prompt_text = {
                "type": "text",
                "text": prefix + _grounding_prompt().format(anno["query"]),
            }

        # --- Student: clean video (或 ZwZ式带标记视频, 或 G1: t2sgrid表示) ---
        if os.environ.get("STUDENT_GRID"):
            # G1 faithful T2SGrid: student自己就用grid表示(train/test都grid)
            student_content = _build_t2sgrid_content(anno["video_path"], self.data_args)
            student_messages = [
                {"role": "user", "content": student_content + [student_prompt_text]}
            ]
        else:
            if self.student_mark_strategy:
                student_vid_path = _create_teacher_video(
                    anno["video_path"], gt_start, gt_end,
                    self.teacher_cache_dir, self.student_mark_strategy,
                )
            else:
                student_vid_path = anno["video_path"]
            student_video = _build_video_content(student_vid_path, self.data_args)
            student_messages = [
                {"role": "user", "content": [student_video, student_prompt_text]}
            ]
        student_inputs, prompt_text = _process_video_inputs(
            student_messages, self.processor,
        )
        student_inputs["input_ids"] = student_inputs["input_ids"][0]
        # Qwen3-VL: mm_token_type_ids 与 input_ids 同步去掉 batch 维
        if "mm_token_type_ids" in student_inputs and student_inputs["mm_token_type_ids"].dim() > 1:
            student_inputs["mm_token_type_ids"] = student_inputs["mm_token_type_ids"][0]

        # --- Teacher: GT-marked video ---
        try:
            if self.teacher_strategy.startswith("timemarker"):
                visual_sub = None
                if "numberit+red_border" in self.teacher_strategy:
                    visual_sub = "numberit+red_border"
                teacher_content = _build_timemarker_content(
                    anno["video_path"], gt_start, gt_end,
                    self.data_args, visual_strategy=visual_sub,
                )
                teacher_content.append(teacher_prompt_text)
                teacher_messages = [{"role": "user", "content": teacher_content}]
            elif self.teacher_strategy == "zoom":
                teacher_content = _build_zoom_content(
                    anno["video_path"], gt_start, gt_end, self.data_args,
                )
                teacher_content.append(teacher_prompt_text)
                teacher_messages = [{"role": "user", "content": teacher_content}]
            elif self.teacher_strategy == "hifps":
                # 均匀高fps teacher: 整段按 TEACHER_HIFPS 采样(默认3), 单video, 时间编码正确。
                _hifps = float(os.environ.get("TEACHER_HIFPS", "3"))
                teacher_video = _build_hifps_content(anno["video_path"], self.data_args, _hifps)
                teacher_messages = [
                    {"role": "user", "content": [teacher_video, teacher_prompt_text]}
                ]
            elif self.teacher_strategy == "t2sgrid":
                # G2: teacher看T2SGrid表示, student看普通视频(蒸馏)
                teacher_content = _build_t2sgrid_content(anno["video_path"], self.data_args)
                teacher_content.append(teacher_prompt_text)
                teacher_messages = [{"role": "user", "content": teacher_content}]
            else:
                # For coarse_box_multi / coarse_box_random_multi: pass other events from the same video
                _other_spans = None
                if self.teacher_strategy in ("coarse_box_multi", "coarse_box_random_multi"):
                    _current = (round(gt_start, 2), round(gt_end, 2))
                    _all = self._video_all_spans.get(anno["video_path"], set())
                    _other_spans = [list(s) for s in _all if s != _current]
                    # 稳定排序: set 迭代序在不同进程/不同 PYTHONHASHSEED 下可变,
                    # 会导致 cache_key 不一致 → 同一样本在不同卡上渲染出不同 teacher 视频
                    _other_spans = sorted(_other_spans)

                teacher_video_path = _create_teacher_video(
                    anno["video_path"], gt_start, gt_end,
                    self.teacher_cache_dir, self.teacher_strategy,
                    other_spans=_other_spans,
                    zoom_expand=self.zoom_expand,
                    blue_zoom_expand=self.blue_zoom_expand,
                    video_duration=anno.get("duration", None),
                )
                teacher_video = _build_video_content(teacher_video_path, self.data_args)
                teacher_messages = [
                    {"role": "user", "content": [teacher_video, teacher_prompt_text]}
                ]
            teacher_inputs, _ = _process_video_inputs(
                teacher_messages, self.processor,
            )
        except Exception as e:
            _TEACHER_FAIL_COUNT += 1
            # 直接打到 stdout(logging.warning 不进训练日志, 会导致静默白跑)。
            # 首次失败打完整 traceback 便于定位根因; 之后每50次打一次累计数。
            if _TEACHER_FAIL_COUNT == 1:
                print(f"\n[TEACHER-FALLBACK] ⚠️ teacher 构造失败, 回退成 teacher=student "
                      f"(JSD将=0, 该样本无蒸馏)! strategy={self.teacher_strategy} "
                      f"video={anno.get('video_id', '?')}\n  原因: {e}", flush=True)
                traceback.print_exc()
            elif _TEACHER_FAIL_COUNT % 50 == 0:
                print(f"[TEACHER-FALLBACK] 累计哑火 {_TEACHER_FAIL_COUNT} 次 "
                      f"(OK {_TEACHER_OK_COUNT} 次) — 若占比高, 本次训练蒸馏基本无效!", flush=True)
            teacher_inputs = student_inputs
            teacher_failed = True
        else:
            _TEACHER_OK_COUNT += 1
            teacher_failed = False

        # --- Combine outputs ---
        result = dict(student_inputs)
        result["prompt"] = student_messages
        result["prompt_text"] = prompt_text
        result["anno"] = anno
        result["teacher_failed"] = teacher_failed

        # Teacher vision tensors (prefixed)
        for key in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts"):
            if key in teacher_inputs:
                result[f"teacher_{key}"] = teacher_inputs[key]
        # t2sgrid等image型teacher的视觉张量
        for key in ("pixel_values", "image_grid_thw"):
            if key in teacher_inputs:
                result[f"teacher_{key}"] = teacher_inputs[key]
        # Qwen3-VL: teacher 的逐token mm_token_type_ids (与 teacher_input_ids 对齐)
        if "mm_token_type_ids" in teacher_inputs:
            _tmm = teacher_inputs["mm_token_type_ids"]
            result["teacher_mm_token_type_ids"] = _tmm[0] if _tmm.dim() > 1 else _tmm

        # 每样本teacher视觉块数(zoom/timemarker多video、t2sgrid多image时>1)。
        # trainer据此按"每样本多块"正确切分; 单video策略=1。
        _nv = 0
        if "video_grid_thw" in teacher_inputs:
            _nv = int(teacher_inputs["video_grid_thw"].shape[0])
        elif "image_grid_thw" in teacher_inputs:
            _nv = int(teacher_inputs["image_grid_thw"].shape[0])
        result["teacher_num_visuals"] = torch.tensor([max(_nv, 1)], dtype=torch.long)

        # Teacher's own input_ids (may differ from student due to different
        # video token count after re-encoding + different text prefix).
        # Always saved — when teacher_failed, teacher_inputs == student_inputs
        # so these match the fallback visual tensors.
        result["teacher_input_ids"] = teacher_inputs["input_ids"][0]
        result["teacher_attention_mask"] = teacher_inputs["attention_mask"][0]

        return result
