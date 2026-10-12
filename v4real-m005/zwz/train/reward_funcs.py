"""Reward functions for TVG GRPO training.

Based on TimeLens's reward_funcs.py, adapted for Temporal-ZwZ.
"""

import re


def extract_answer(text):
    """Extract content from <answer> tags, or return full text."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return matches[-1].strip() if matches else text.strip()


def extract_time(paragraph):
    """Extract (start, end) timestamps from text. Returns list of tuples."""
    paragraph = paragraph.lower()
    timestamps = []

    time_regex = re.compile(
        r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b"
    )
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]

    if time_matches:
        converted = []
        for t in time_matches:
            parts = t.split(":")
            if len(parts) == 3:
                h, m = int(parts[0]), int(parts[1])
                s = float(parts[2])
                converted.append(h * 3600 + m * 60 + s)
            elif len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                converted.append(m * 60 + s)
        timestamps = [
            (converted[i], converted[i + 1])
            for i in range(0, len(converted), 2)
        ]

    if not timestamps:
        for pattern in [
            r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",
        ]:
            matches = re.findall(pattern, paragraph)
            if matches:
                timestamps = [(float(s), float(e)) for s, e in matches]
                break

    if not timestamps:
        num_matches = re.findall(r"\b(\d+\.\d+|\d+)\b", paragraph)
        num_matches = num_matches[: len(num_matches) // 2 * 2]
        timestamps = [
            (float(num_matches[i]), float(num_matches[i + 1]))
            for i in range(0, len(num_matches), 2)
        ]

    return timestamps


def iou(a, b):
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    return max(min1 - max0, 0) / max(max1 - min0, 1e-8)


def tiou_reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
    """Temporal IoU reward for GRPO.

    Compatible with TimeLens's reward interface.
    """
    pattern = r"<\|(video_pad|image_pad|vision_start|vision_end)\|>"
    prompt_text = [re.sub(pattern, "", text) for text in prompt_text]

    completions = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(c) for c in completions]
    timestamps_list = [extract_time(a) for a in answers]

    rewards = []
    for i, timestamps in enumerate(timestamps_list):
        gt = anno[i]["span"]
        if isinstance(gt[0], list):
            gt = gt[0]

        if not timestamps:
            rewards.append(0.0)
            continue

        pred = timestamps[-1]
        rewards.append(iou(pred, gt))

    return rewards


def tiou_dist_reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
    """tIoU + 距离塑形: 没overlap时按"离GT多近"给分, 救活GRPO死区(全错样本)。
    reward = tIoU + beta * (1 - 边界距离/时长)。beta 由 REWARD_DIST_BETA 控制(默认0.2)。
    """
    import os
    beta = float(os.environ.get("REWARD_DIST_BETA", "0.2"))
    pattern = r"<\|(video_pad|image_pad|vision_start|vision_end)\|>"
    prompt_text = [re.sub(pattern, "", text) for text in prompt_text]
    completions = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(c) for c in completions]
    timestamps_list = [extract_time(a) for a in answers]

    rewards = []
    for i, timestamps in enumerate(timestamps_list):
        gt = anno[i]["span"]
        if isinstance(gt[0], list):
            gt = gt[0]
        if not timestamps:
            rewards.append(0.0)
            continue
        pred = timestamps[-1]
        t = iou(pred, gt)
        dur = anno[i].get("duration", 0) or 0
        if dur > 0:
            # 边界距离(start+end平均), 归一化到[0,1]; 顺便压边界精度(R1@0.7)
            d = (abs(pred[0] - gt[0]) + abs(pred[1] - gt[1])) / (2.0 * dur)
            d = min(d, 1.0)
            rewards.append(t + beta * (1.0 - d))
        else:
            rewards.append(t)
    return rewards


def tiou_seg_reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
    """segment级边界奖励: tIoU + 对**精确命中**的边界给奖励(SDPO思路)。
    与失败的距离塑形不同: 距离塑形奖励"大致接近"(导致蒙混饱和);
    这里只在 start/end 各自**误差<紧阈值τ**时给bonus → 逼精度(打R1@0.7)。
    γ=REWARD_SEG_GAMMA(默认0.2), τ=REWARD_SEG_TAU*duration(默认0.05, 即5%时长)。
    """
    import os
    gamma = float(os.environ.get("REWARD_SEG_GAMMA", "0.2"))
    tau_frac = float(os.environ.get("REWARD_SEG_TAU", "0.05"))
    pattern = r"<\|(video_pad|image_pad|vision_start|vision_end)\|>"
    prompt_text = [re.sub(pattern, "", text) for text in prompt_text]
    completions = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(c) for c in completions]
    timestamps_list = [extract_time(a) for a in answers]

    rewards = []
    for i, timestamps in enumerate(timestamps_list):
        gt = anno[i]["span"]
        if isinstance(gt[0], list):
            gt = gt[0]
        if not timestamps:
            rewards.append(0.0)
            continue
        pred = timestamps[-1]
        t = iou(pred, gt)
        dur = anno[i].get("duration", 0) or 0
        if dur > 0:
            tau = tau_frac * dur
            hit_s = 1.0 if abs(pred[0] - gt[0]) < tau else 0.0
            hit_e = 1.0 if abs(pred[1] - gt[1]) < tau else 0.0
            rewards.append(t + gamma * (hit_s + hit_e) / 2.0)
        else:
            rewards.append(t)
    return rewards


def tiou_anchor_reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
    """Grounded-CoT reward (TAR-TVG式): 最终tIoU + "渐进anchor单调收敛"奖励。

    要求模型在推理中输出多条 'estimate: X - Y' 中间估计, 末行给最终答案。
    奖励 = tIoU(最终, GT)  +  γ · 收敛奖励
      收敛奖励 = clip(tIoU(最终) − tIoU(第一个anchor), 0, 1)
                (推理过程把估计【改好了】才给分; 全程不变=0, 不惩罚)
    目的: 逼CoT做真正的"渐进定位"而非free-form漂移(后者TVG上有害)。
    γ 由 REWARD_ANCHOR_GAMMA 控制(默认0.3)。
    """
    import os
    gamma = float(os.environ.get("REWARD_ANCHOR_GAMMA", "0.3"))
    completions = [c[0]["content"] for c in completions]
    rewards = []
    for i, c in enumerate(completions):
        gt = anno[i]["span"]
        if isinstance(gt[0], list):
            gt = gt[0]
        # 最终答案: 复用 extract_answer+extract_time, 取最后一对
        final_ts = extract_time(extract_answer(c))
        base = iou(final_ts[-1], gt) if final_ts else 0.0
        # 中间anchor: 所有 'estimate: X - Y'
        anchors = re.findall(r"estimate:?\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", c.lower())
        bonus = 0.0
        if len(anchors) >= 2:
            first_iou = iou((float(anchors[0][0]), float(anchors[0][1])), gt)
            improved = base - first_iou          # 推理把估计改好了多少
            bonus = gamma * max(0.0, min(improved, 1.0))
        rewards.append(base + bonus)
    return rewards


def tiou_sharp_reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
    """指标对齐的精度reward: tIoU + 跨阈值奖励(直接优化 R@0.3/0.5/0.7)。
    把梯度压力压到"跨过0.5/0.7"上→直冲IoU@0.7精度。阈值奖励由env控制。
    """
    import os
    b3 = float(os.environ.get("REWARD_SHARP_B3", "0.1"))
    b5 = float(os.environ.get("REWARD_SHARP_B5", "0.2"))
    b7 = float(os.environ.get("REWARD_SHARP_B7", "0.3"))
    completions = [c[0]["content"] for c in completions]
    rewards = []
    for i, c in enumerate(completions):
        gt = anno[i]["span"]
        if isinstance(gt[0], list):
            gt = gt[0]
        ts = extract_time(extract_answer(c))
        t = iou(ts[-1], gt) if ts else 0.0
        r = t + (b3 if t >= 0.3 else 0.0) + (b5 if t >= 0.5 else 0.0) + (b7 if t >= 0.7 else 0.0)
        rewards.append(r)
    return rewards


def tiou_pow_reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
    """精度reward(幂): tIoU^k(k>1, 默认2)。近1更陡→逼边界更准; 近0更平→不奖励粗略重叠。"""
    import os
    k = float(os.environ.get("REWARD_POW_K", "2.0"))
    completions = [c[0]["content"] for c in completions]
    rewards = []
    for i, c in enumerate(completions):
        gt = anno[i]["span"]
        if isinstance(gt[0], list):
            gt = gt[0]
        ts = extract_time(extract_answer(c))
        t = iou(ts[-1], gt) if ts else 0.0
        rewards.append(t ** k)
    return rewards


def tiou_softrange_reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
    """软边界reward(模型自报不确定性): 用 SOFTRANGE_MODE prompt, 模型输出
    'The event happens in X - Y seconds.' + 'uncertainty: u seconds'。
    reward = tIoU(中心) + β·[GT落在[X-u, Y+u]内(校准命中)] − δ·(u/GT跨度)(过宽惩罚)。
    → 逼模型学"又窄又覆盖GT"的校准边界 = soft boundary, 直冲精度。
    u缺省=0时退化成纯tIoU(安全)。
    """
    import os
    beta = float(os.environ.get("REWARD_SR_BETA", "0.3"))
    delta = float(os.environ.get("REWARD_SR_DELTA", "0.1"))
    completions = [c[0]["content"] for c in completions]
    rewards = []
    for i, c in enumerate(completions):
        gt = anno[i]["span"]
        if isinstance(gt[0], list):
            gt = gt[0]
        gs, ge = float(gt[0]), float(gt[1])
        m = re.findall(r"happens in\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", c.lower())
        if not m:
            rewards.append(0.0); continue
        s, e = float(m[-1][0]), float(m[-1][1])
        base = iou((s, e), (gs, ge))
        um = re.findall(r"uncertainty:?\s*(\d+\.?\d*)", c.lower())
        u = float(um[-1]) if um else 0.0
        covered = (s - u <= gs) and (ge <= e + u)
        span = max(ge - gs, 1.0)
        reward = base + (beta if (covered and u > 0) else 0.0) - delta * (u / span)
        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs):
    """Check <think>...</think><answer>...</answer> format."""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    contents = [c[0]["content"] for c in completions]
    return [1.0 if re.fullmatch(pattern, c.strip()) else 0.0 for c in contents]


REWARD_REGISTRY = {
    "tiou": tiou_reward,
    "tiou_dist": tiou_dist_reward,
    "tiou_seg": tiou_seg_reward,
    "tiou_anchor": tiou_anchor_reward,
    "tiou_sharp": tiou_sharp_reward,
    "tiou_pow": tiou_pow_reward,
    "tiou_softrange": tiou_softrange_reward,
    "format": format_reward,
}


def load_reward_funcs(names):
    """Load reward functions by name."""
    funcs = []
    for name in names:
        if name not in REWARD_REGISTRY:
            raise ValueError(f"Unknown reward: {name}. Available: {list(REWARD_REGISTRY)}")
        funcs.append(REWARD_REGISTRY[name])
    return funcs
