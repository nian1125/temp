"""OPD-GRPO Trainer: GRPO with On-Policy Distillation for TVG.

Extends TimeLens's QwenvlGRPOTrainer to add teacher-student distillation.
The teacher sees GT-marked video, the student sees clean video.
Same model weights, different visual inputs.

Distillation uses Generalized JSD with top-k logit filtering,
following Vision-OPD's compute_self_distillation_loss.
"""

import logging
import math
import os
import sys
from typing import Any, Union

import torch
import torch.nn.functional as F
from trl.trainer.utils import selective_log_softmax

from zwz.train.reward_funcs import extract_time, iou

_timelens_root = os.environ.get("TIMELENS_ROOT")
if _timelens_root and _timelens_root not in sys.path:
    sys.path.insert(0, _timelens_root)

from training.trainer.grpo_trainer_qwenvl import (
    QwenvlGRPOTrainer,
    QWEN_2_5_VL_KEY_WORDS,
)

TEACHER_VISION_KEYS = tuple(f"teacher_{k}" for k in QWEN_2_5_VL_KEY_WORDS)
TEACHER_TENSOR_KEYS = tuple(
    f"teacher_{k}" for k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")
) + ("teacher_num_visuals",)
TEACHER_LIST_KEYS = ("teacher_second_per_grid_ts",)


def _compute_jsd_topk(student_logits, teacher_logits, temperature, alpha, topk, completion_mask,
                      return_per_sample=False, divergence="jsd", clip_max=0.0,
                      entropy_gate="off", entropy_tau=0.8,
                      focus_mask=None, focus_mode="off", focus_weight=4.0):
    """Compute distillation divergence with top-k logit filtering.

    divergence:
      - "jsd": Generalized JSD (alpha 插值, Vision-OPD 默认)
      - "fkl": Forward KL = KL(teacher || student), mass-covering (OPSD 实测最优)
      - "rkl": Reverse KL = KL(student || teacher), mode-seeking
      - "tvd": Total Variation Distance = 0.5 * sum|P - Q|, 对称有界 [0,1]
      - "symkl": Symmetric KL = KL(T||S) + KL(S||T), Jeffreys divergence
    clip_max>0: 对每个 token 的散度值截断上限, 防重尾token梯度爆炸 (REOPOLD/OPSD 稳定化)

    focus_mode (token 级聚焦, 解决"模板token稀释答案token"):
      - "off"      : 原行为, 在全部 completion token 上均摊
      - "numeric"  : 硬聚焦, 只在 focus_mask=1 的 token(数值token)上蒸馏并只在其上取平均
      - "numeric_soft": 软聚焦, focus token 权重=focus_weight, 其余=1, 每样本归一化使均值≈1
    focus_mask: (B, L) float, 1=需要聚焦的 token。为 None 时 focus_mode 自动失效。
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)

    if topk is not None and topk < student_logits.size(-1):
        # Select top-k indices from teacher distribution
        _, topk_indices = teacher_log_probs.topk(topk, dim=-1)  # (B, L, k)

        # Gather top-k log probs
        s_topk = student_log_probs.gather(-1, topk_indices)  # (B, L, k)
        t_topk = teacher_log_probs.gather(-1, topk_indices)  # (B, L, k)

        # Add tail bucket: log(1 - sum(top_k_probs))
        def add_tail(log_probs):
            log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_s = torch.clamp(log_s, max=-1e-7)
            tail_log = torch.log(-torch.expm1(log_s))
            return torch.cat([log_probs, tail_log], dim=-1)

        s_distill = add_tail(s_topk)  # (B, L, k+1)
        t_distill = add_tail(t_topk)  # (B, L, k+1)
    else:
        s_distill = student_log_probs
        t_distill = teacher_log_probs

    # Divergence 选择
    if divergence == "fkl":
        # Forward KL = KL(teacher || student): mass-covering
        per_token_jsd = F.kl_div(s_distill, t_distill, reduction="none", log_target=True).sum(-1)
    elif divergence == "rkl":
        # Reverse KL = KL(student || teacher): mode-seeking
        per_token_jsd = F.kl_div(t_distill, s_distill, reduction="none", log_target=True).sum(-1)
    elif divergence == "tvd":
        # Total Variation Distance = 0.5 * sum|P - Q|, 对称有界
        s_probs = s_distill.exp()
        t_probs = t_distill.exp()
        per_token_jsd = 0.5 * (t_probs - s_probs).abs().sum(-1)
    elif divergence == "symkl":
        # Symmetric KL = KL(T||S) + KL(S||T), Jeffreys divergence
        fkl = F.kl_div(s_distill, t_distill, reduction="none", log_target=True).sum(-1)
        rkl = F.kl_div(t_distill, s_distill, reduction="none", log_target=True).sum(-1)
        per_token_jsd = fkl + rkl
    else:
        # Generalized JSD (默认)
        alpha_t = torch.tensor(alpha, dtype=s_distill.dtype, device=s_distill.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([
                s_distill + torch.log(1 - alpha_t),
                t_distill + torch.log(alpha_t),
            ]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, t_distill, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, s_distill, reduction="none", log_target=True)
        per_token_jsd = torch.lerp(kl_student, kl_teacher, alpha_t).sum(-1)  # (B, L)

    # 重尾截断 (稳定化): 防止单个token的散度→∞主导梯度
    if clip_max and clip_max > 0:
        per_token_jsd = per_token_jsd.clamp(max=float(clip_max))

    # Entropy-gating: 按 teacher 逐token熵重分配蒸馏权重(聚焦高熵=边界决策token)
    denom_mask = completion_mask
    if entropy_gate != "off":
        t_probs = t_distill.exp()
        H_t = -(t_probs * t_distill).sum(-1)  # (B,L) teacher top-k+tail 熵
        if entropy_gate == "soft":
            # 每样本归一化使权重均值≈1(总量守恒, 仅在token间重分配)
            wsum = (H_t * completion_mask).sum(-1, keepdim=True)
            wcnt = completion_mask.sum(-1, keepdim=True).clamp(min=1.0)
            wmean = (wsum / wcnt).clamp(min=1e-6)
            per_token_jsd = per_token_jsd * (H_t / wmean)
        elif entropy_gate == "hard":
            gate = (H_t > entropy_tau).float()
            per_token_jsd = per_token_jsd * gate
            denom_mask = completion_mask * gate  # 只在高熵token上平均

    # Token 聚焦 (numeric focus): completion 里 ~一半 token 是固定模板("The event happens in ... seconds."),
    # 师生必然一致, 在全token均摊会把真正携带答案的数值token信号稀释掉。这里把散度压到数值token上。
    if focus_mode != "off" and focus_mask is not None:
        fm = focus_mask.to(per_token_jsd.dtype)
        if focus_mode == "numeric":
            # 保底: 若某样本一个数值token都没有(解析失败/空输出), 回退到全token, 避免白丢样本
            _empty = (fm.sum(-1, keepdim=True) <= 0)
            fm = torch.where(_empty.expand_as(fm), denom_mask.to(fm.dtype), fm)
            per_token_jsd = per_token_jsd * fm
            denom_mask = fm
        elif focus_mode == "numeric_soft":
            w = 1.0 + (float(focus_weight) - 1.0) * fm
            # 每样本归一化, 使权重均值≈1 → 只在token间重分配, 不改变JSD总量级
            wsum = (w * completion_mask).sum(-1, keepdim=True)
            wcnt = completion_mask.sum(-1, keepdim=True).clamp(min=1.0)
            w = w / (wsum / wcnt).clamp(min=1e-6)
            per_token_jsd = per_token_jsd * w

    # Masked mean
    per_sample_jsd = (
        (per_token_jsd * completion_mask).sum(-1)
        / denom_mask.sum(-1).clamp(min=1.0)
    )
    jsd_loss = per_sample_jsd.mean()
    if return_per_sample:
        return jsd_loss, per_sample_jsd
    return jsd_loss


def _split_visual_kwargs(kwargs, start, end):
    """Split packed visual inputs for batch slice [start, end)."""
    result = {}
    for key, val in kwargs.items():
        if key in ('video_grid_thw', 'image_grid_thw'):
            result[key] = val[start:end]
        elif key == 'pixel_values_videos' and 'video_grid_thw' in kwargs:
            thw = kwargs['video_grid_thw']
            cum = thw.prod(dim=-1).cumsum(0)
            s = 0 if start == 0 else int(cum[start - 1].item())
            e = int(cum[end - 1].item())
            result[key] = val[s:e]
        elif key == 'pixel_values' and 'image_grid_thw' in kwargs:
            thw = kwargs['image_grid_thw']
            cum = thw.prod(dim=-1).cumsum(0)
            s = 0 if start == 0 else int(cum[start - 1].item())
            e = int(cum[end - 1].item())
            result[key] = val[s:e]
        elif isinstance(val, list):
            result[key] = val[start:end]
        elif isinstance(val, torch.Tensor):
            result[key] = val[start:end]
        else:
            result[key] = val
    return result


class OPDGRPOTrainer(QwenvlGRPOTrainer):
    """GRPO trainer with On-Policy Distillation.

    Loss = L_grpo + kl_weight * L_jsd

    The teacher receives GT-marked video input; the student sees clean video.
    Teacher forward is always done with torch.no_grad().

    Teacher weights:
      - frozen_teacher is None (默认, exp1式): teacher = 正在训练的 student 本体,
        权重随 student 每步一起更新 (self-distillation, 共享权重)。
      - frozen_teacher 给定 (exp4式): teacher = 独立冻结的初始 SFT 模型,
        全程不更新, 只有 student 被优化。

    Distillation uses Generalized JSD with top-k (Vision-OPD style).
    """

    def __init__(
        self,
        *args,
        kl_weight: float = 1.0,
        grpo_weight: float = 1.0,
        kl_temperature: float = 1.0,
        jsd_alpha: float = 0.5,
        distillation_topk: int = 100,
        teacher_quality_gate: bool = False,
        kl_weight_end: float = None,
        kl_schedule: str = "constant",
        divergence: str = "jsd",
        jsd_clip_max: float = 0.0,
        exopd_lambda: float = 1.25,
        entropy_gate: str = "off",
        entropy_tau: float = 0.8,
        golden_keep_frac: float = 1.0,
        two_stage_explore_frac: float = 0.0,
        exopd_ref_path: str = None,
        frozen_teacher=None,
        teacher_update_strategy: str = "drifting",
        ema_decay: float = 0.0,
        dead_rescue_enabled: bool = False,
        dead_rescue_mode: str = "fixed",
        dead_reward_std_eps: float = 1e-6,
        dead_reward_mean_threshold: float = 0.30,
        dead_group_criterion: str = "std_zero_lowmean",
        teacher_advantage_margin: float = 0.05,
        dead_rescue_multiplier: float = 2.0,
        negative_teacher_multiplier: float = 0.25,
        # --- 以下为新增开关, 默认值均复刻原行为, 不影响已有实验 ---
        teacher_tiou_mode: str = "argmax",
        teacher_gen_max_new_tokens: int = 48,
        teacher_gate_mode: str = "off",
        dead_rescue_norm: str = "weighted",
        jsd_token_focus: str = "off",
        jsd_numeric_weight: float = 4.0,
        adaptive_controller: str = "off",
        controller_lambda_no_edge: float = 0.0,
        controller_lambda_dead: float = 2.0,
        controller_skip_zero: bool = True,
        teacher_rollout_mode: str = "sample",
        teacher_rollout_n: int = 0,
        teacher_rollout_batch: int = 0,
        **kwargs,
    ):
        # 修复: 父类(TimeLens GRPO)在 beta>0 时用 AutoModelForCausalLM 加载 reference,
        # 但 Qwen3-VL 是 VLM, 需 AutoModelForImageTextToText。临时替换父类模块里的该名字,
        # 让 ref 用正确的类加载(主模型已实例化传入, 不走该路径, 故安全)。
        import training.trainer.grpo_trainer_qwenvl as _pmod
        from transformers import AutoModelForImageTextToText as _ITT
        _orig_automodel = getattr(_pmod, "AutoModelForCausalLM", None)
        _pmod.AutoModelForCausalLM = _ITT
        try:
            super().__init__(*args, **kwargs)
        finally:
            if _orig_automodel is not None:
                _pmod.AutoModelForCausalLM = _orig_automodel
        self.kl_weight = kl_weight
        self.grpo_weight = grpo_weight
        self.kl_temperature = kl_temperature
        self.jsd_alpha = jsd_alpha
        self.distillation_topk = distillation_topk
        self.teacher_quality_gate = teacher_quality_gate
        self.divergence = divergence
        self.jsd_clip_max = jsd_clip_max
        self.exopd_lambda = exopd_lambda
        self.entropy_gate = entropy_gate
        self.entropy_tau = entropy_tau
        self.golden_keep_frac = golden_keep_frac
        self.two_stage_explore_frac = float(two_stage_explore_frac)

        # 冻结 teacher (exp4): 独立的初始 SFT 模型, 全程不更新, 只做特权视频前向。
        # None = 默认 self-distillation (teacher=student本体, 权重共享随训练更新, exp1)。
        # DeepSpeed 只 prepare self.model, frozen_teacher 不被包裹 → 首次前向时惰性搬到
        # student 所在 device 并对齐 dtype (见 _get_teacher_module)。
        self.frozen_teacher = frozen_teacher
        if self.frozen_teacher is not None:
            self.frozen_teacher.eval()
            for p in self.frozen_teacher.parameters():
                p.requires_grad_(False)
            self.frozen_teacher.config.use_cache = False
        self._frozen_teacher_ready = False

        # EMA teacher: 每步 optimizer 后 EMA 更新 student 权重副本
        # ema_decay=0 禁用; >0 时 teacher 用 EMA 权重而非当前 student 权重
        self.ema_decay = ema_decay
        self._ema_model = None  # lazy init, deepcopy of model
        self._ema_ready = False

        # Teacher update strategy: "drifting" (默认, 每步用当前权重) |
        # "cached_snapshot" (每 rollout block 快照一次, 4次更新共享同一teacher logits)
        self.teacher_update_strategy = teacher_update_strategy
        self._cached_teacher_logits = None  # 快照缓存 (B, L, vocab)
        self._cached_teacher_checksum = None  # 缓存校验
        self._cached_block_id = -1

        # Dead-group rescue: reward 分布死组(所有completion reward接近且低)时增强JSD
        # dead_rescue_mode: "fixed"=修复版(shuffle前整组检测+逐样本加权) |
        #                   "legacy"=Run B/G时代行为(textual_logs检测能触发, 但×平均权重, 死组无相对放大)
        self.dead_rescue_enabled = dead_rescue_enabled
        self.dead_rescue_mode = dead_rescue_mode
        self.dead_reward_std_eps = dead_reward_std_eps
        self.dead_reward_mean_threshold = dead_reward_mean_threshold
        # dead group 判据: "std_zero_lowmean"(原行为) | "all_zero"(论文 r_{g,i}=0 ∀i)
        self.dead_group_criterion = dead_group_criterion
        self.teacher_advantage_margin = teacher_advantage_margin
        self.dead_rescue_multiplier = dead_rescue_multiplier
        self.negative_teacher_multiplier = negative_teacher_multiplier

        # ================= 新增开关 (默认值 = 原行为, 老实验可复现) =================
        # [修3a] teacher tIoU 估计方式:
        #   "argmax"   : 原行为 — 在 student 采样出的前缀上取 teacher teacher-forced argmax。
        #                数字逐位生成, 前缀被student锚定 → 系统性偏向student答案, 只能当粗代理。
        #   "generate" : teacher 用特权视频【自主贪心解码】, 每个 rollout block 只算一次,
        #                结果随统一 shuffle/split 对齐到各切片(与 dead_mask 同一条路径)。
        self.teacher_tiou_mode = teacher_tiou_mode
        self.teacher_gen_max_new_tokens = int(teacher_gen_max_new_tokens)
        self._teacher_gen_warned = False

        # [修3b] teacher 质量门控开关。父类现在会返回逐completion的 "rewards",
        # 原代码 inputs.get("rewards") 恒为 None 导致该分支是死代码。为不改变老实验结果,
        # 这里默认 "off" = 即使 rewards 可用也不进该分支; 显式设 "on" 才启用。
        self.teacher_gate_mode = teacher_gate_mode

        # [修2] dead-group rescue 的归一化方式:
        #   "weighted" : 原行为 sum(j*w)/sum(w) — 当 loss batch=1 时 w 被自己的分母除掉,
        #                恒等变换, rescue 完全无效(这是主实验里 rescue 哑火的根因)。
        #   "mean"     : sum(j*w)/B — 死组样本的 JSD 被真正放大 multiplier 倍。
        #                B=1 时等价于"死组步提高 kl_weight", B>1 时是组内相对放大。
        self.dead_rescue_norm = dead_rescue_norm

        # [修4] JSD token 聚焦: off | numeric(硬) | numeric_soft(软加权)
        self.jsd_token_focus = jsd_token_focus
        self.jsd_numeric_weight = float(jsd_numeric_weight)
        self._numeric_tok_cache = {}

        # [论文 Eq5] 组级 adaptive controller
        #   "off"   : 原行为 — batch级 teacher_advantage 标量缩放, 且仅在 dead_rescue 内生效
        #   "paper" : λ_g ∈ {no_edge, 1, dead} 逐组判定, 论文 Eq5/Eq6 口径
        self.adaptive_controller = adaptive_controller
        self.controller_lambda_no_edge = float(controller_lambda_no_edge)  # 论文: 0
        self.controller_lambda_dead = float(controller_lambda_dead)        # 论文: 2
        # λ 全 0 的切片直接跳过 teacher forward (纯省算力, 不改变数学结果)
        self.controller_skip_zero = bool(controller_skip_zero)
        self._ctrl_warned = False
        self._last_teacher_group_tiou = {}

        # teacher 独立 rollout 设置 (controller 的 r̄^T 来源)
        #   "sample": 论文口径 — 与 student 同 temperature/top_p 采样 teacher_rollout_n 条
        #   "greedy": 贪心 1 条 (便宜, 但与 student 采样均值不可直接比较, 会高估 teacher)
        self.teacher_rollout_mode = teacher_rollout_mode
        self.teacher_rollout_n = int(teacher_rollout_n)      # 0 → 用 num_generations (=G)
        self.teacher_rollout_batch = int(teacher_rollout_batch)  # 0 → 一次全量; >0 分批省显存
        # =========================================================================

        # ExOPD: ref = 生成时的 student 自身 (stop-grad), 不再加载独立模型 → 不爆显存。
        # (π-Distill App D.2: 用 πθ 作 reference 省显存且更稳。)
        # 注意前提: ExOPD 是"超越teacher", 仅在 teacher>student(正Δ)时有意义;
        # numberit teacher 弱于 student → 应配 coarse_box 等正Δ teacher。
        self.exopd_ref = None
        # kl_weight schedule: 前低后高(或任意线性). constant=用固定 kl_weight.
        self.kl_weight_start = kl_weight
        self.kl_weight_end = kl_weight if kl_weight_end is None else kl_weight_end
        self.kl_schedule = kl_schedule
        self._metrics["train"]["jsd_distill"] = []
        self._metrics["eval"]["jsd_distill"] = []
        self._metrics["train"]["grpo_loss"] = []
        self._metrics["eval"]["grpo_loss"] = []
        self._metrics["train"]["weighted_grpo"] = []
        self._metrics["eval"]["weighted_grpo"] = []
        self._metrics["train"]["weighted_jsd"] = []
        self._metrics["eval"]["weighted_jsd"] = []
        self._metrics["train"]["teacher_student_top1_agree"] = []
        self._metrics["eval"]["teacher_student_top1_agree"] = []
        if teacher_quality_gate:
            self._metrics["train"]["teacher_tiou"] = []
            self._metrics["eval"]["teacher_tiou"] = []
            self._metrics["train"]["jsd_gate_mean"] = []
            self._metrics["eval"]["jsd_gate_mean"] = []

        # --- 诊断指标 (Run B 增强) ---
        # 无论是否启用 dead_rescue / cached_snapshot, 都记录这些指标便于分析
        for diag_key in [
            "teacher_tiou_mean",          # teacher argmax 解码的 tIoU 均值
            "student_reward_mean",        # student rollout 的 reward 均值
            "teacher_advantage_mean",     # teacher_tiou - student_reward 均值
            "teacher_better_frac",        # teacher 优于 student 的样本比例
            "teacher_worse_frac",         # teacher 差于 student 的样本比例
            "jsd_start_raw",              # JSD 在加权前的原始值 (block 首步)
            "jsd_end_raw",                # JSD 在加权前的原始值 (block 末步)
            "jsd_total_raw",              # JSD 加权前原始均值
            "jsd_total_weighted",         # JSD 加权后均值 (= weighted_jsd)
            "sample_jsd_weight",          # 每样本 JSD 权重均值
            "dead_rescue_frac",           # 死组样本占比
            "dead_rescue_triggered",      # 本步是否触发了 dead rescue
            "negative_teacher_downweighted",  # teacher 差于 student 时 JSD 衰减触发
            "update_idx_in_block",        # 当前更新在 rollout block 中的序号
            "generation_block_id",         # 当前 rollout block ID
            "teacher_cache_reused",       # 本步是否复用了缓存的 teacher logits
            "teacher_tiou_gen_mean",      # [新] teacher 自主解码 tIoU 均值 (teacher_tiou_mode=generate)
            "teacher_gate_applied",       # [新] teacher 质量门控是否实际改动了 loss
            "jsd_focus_frac",             # [新] 参与蒸馏的 token 占比 (numeric focus 生效度)
            "ctrl_lambda_mean",           # [新] 论文Eq5 λ_g 均值
            "ctrl_lambda_zero_frac",      # [新] λ_g=0 的样本占比 (teacher 无优势被屏蔽的比例)
            "ctrl_lambda_dead_frac",      # [新] λ_g=dead 的样本占比 (all-zero 且 teacher 有优势)
            "ctrl_skip_teacher",          # [新] 因 λ 全0 跳过 teacher forward 的步数占比
            "teacher_rollout_tiou",       # [新] teacher 独立 rollout 的 r̄^T
        ]:
            self._metrics["train"][diag_key] = []
            self._metrics["eval"][diag_key] = []

        # 逐样本诊断日志
        import os as _os
        self._diag_jsonl_path = None  # 由 train_opd_tvg.py 设置

    _OPD_METRIC_KEYS = (
        "jsd_distill", "grpo_loss", "weighted_grpo", "weighted_jsd",
        "teacher_student_top1_agree", "teacher_tiou", "jsd_gate_mean",
        # 诊断指标
        "teacher_tiou_mean", "student_reward_mean", "teacher_advantage_mean",
        "teacher_better_frac", "teacher_worse_frac",
        "jsd_start_raw", "jsd_end_raw", "jsd_total_raw", "jsd_total_weighted",
        "sample_jsd_weight", "dead_rescue_frac", "dead_rescue_triggered",
        "negative_teacher_downweighted", "update_idx_in_block",
        "generation_block_id", "teacher_cache_reused",
        "teacher_tiou_gen_mean", "teacher_gate_applied", "jsd_focus_frac",
        "ctrl_lambda_mean", "ctrl_lambda_zero_frac", "ctrl_lambda_dead_frac",
        "ctrl_skip_teacher", "teacher_rollout_tiou",
    )

    def _get_teacher_module(self, model):
        """返回执行 teacher 前向的模块。

        优先级: ema_model > frozen_teacher > student本体
        """
        _student_inner = model.module if hasattr(model, "module") else model
        # EMA teacher: 用 EMA 权重副本
        if self.ema_decay > 0 and self._ema_model is not None:
            if not self._ema_ready:
                ref_param = next(_student_inner.parameters())
                self._ema_model.to(device=ref_param.device, dtype=ref_param.dtype)
                self._ema_model.eval()
                self._ema_ready = True
            return self._ema_model
        # Frozen teacher (exp4): 独立冻结模型
        if self.frozen_teacher is not None:
            if not self._frozen_teacher_ready:
                ref_param = next(_student_inner.parameters())
                self.frozen_teacher.to(device=ref_param.device, dtype=ref_param.dtype)
                self.frozen_teacher.eval()
                self._frozen_teacher_ready = True
            return self.frozen_teacher
        # Self-distillation (默认): teacher = student 本体
        return _student_inner

    def _update_ema(self):
        """每步 optimizer.step() 后更新 EMA teacher 权重。

        ema = decay * ema + (1-decay) * student
        首次调用时 deepcopy student 作为初始 EMA。
        """
        if self.ema_decay <= 0:
            return
        _model = self.accelerator.unwrap_model(self.model) if hasattr(self, 'accelerator') else self.model
        _inner = _model.module if hasattr(_model, 'module') else _model
        if self._ema_model is None:
            import copy as _copy
            self._ema_model = _copy.deepcopy(_inner)
            for p in self._ema_model.parameters():
                p.requires_grad_(False)
            self._ema_model.eval()
            self._ema_model.config.use_cache = False
            self._ema_ready = False
            return
        with torch.no_grad():
            for ema_p, p in zip(self._ema_model.parameters(), _inner.parameters()):
                if p.device != ema_p.device:
                    p = p.to(ema_p.device)
                ema_p.mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch,
                                  ignore_keys_for_eval, start_time, learning_rate=None):
        """override: optimizer.step() 后先更新 EMA teacher, 再走父类逻辑。"""
        self._update_ema()
        super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch,
                                          ignore_keys_for_eval, start_time, learning_rate)

    def _current_kl_weight(self):
        """按训练进度返回有效 kl_weight。constant=固定; linear=从start线性到end。"""
        if self.kl_schedule == "constant" or self.kl_weight_end == self.kl_weight_start:
            return self.kl_weight_start
        max_steps = getattr(self.state, "max_steps", 0) or getattr(self.args, "max_steps", 0)
        if not max_steps or max_steps <= 0:
            return self.kl_weight_start
        frac = min(max(self.state.global_step / max_steps, 0.0), 1.0)
        return self.kl_weight_start + (self.kl_weight_end - self.kl_weight_start) * frac

    # ------------------------------------------------------------------
    # [修4] numeric token focus: 定位 completion 里真正携带答案的数值 token
    # ------------------------------------------------------------------
    def _tokenizer(self):
        return getattr(self.processing_class, "tokenizer", self.processing_class)

    def _build_numeric_focus_mask(self, completion_ids, completion_mask):
        """返回 (B, L) float mask: 1=数值token(含数字或小数点), 0=其他(模板词/标点)。

        completion 平均只有 ~18 token, 其中约一半是 'The event happens in ... seconds.'
        这类固定模板, 师生必然一致 → 在全 token 上均摊会把答案token的散度稀释掉。
        逐 token 文本判定结果按 token_id 缓存, 摊销后几乎零开销。
        """
        tok = self._tokenizer()
        uniq = torch.unique(completion_ids).tolist()
        for t in uniq:
            t = int(t)
            if t not in self._numeric_tok_cache:
                try:
                    s = tok.decode([t])
                except Exception:
                    s = ""
                self._numeric_tok_cache[t] = bool(
                    any(c.isdigit() for c in s) or s.strip() in (".", ".-", "-")
                )
        lut_size = int(max(uniq)) + 1 if uniq else 1
        lut = torch.zeros(lut_size, dtype=torch.bool, device=completion_ids.device)
        hit = [int(t) for t in uniq if self._numeric_tok_cache[int(t)]]
        if hit:
            lut[torch.tensor(hit, dtype=torch.long, device=completion_ids.device)] = True
        fm = lut[completion_ids.clamp(max=lut_size - 1)].to(completion_mask.dtype)
        return fm * completion_mask

    # ------------------------------------------------------------------
    # teacher rollout: 论文 §Adaptive controller 的 r̄_g^T
    # ------------------------------------------------------------------
    @staticmethod
    def _group_key(d):
        """rollout group 的唯一标识 = (video, query, gt_span)。"""
        anno = d.get("anno", {}) or {}
        span = anno.get("span", [[0.0, 0.0]])
        if isinstance(span[0], list):
            span = span[0]
        return (anno.get("video_path", ""), anno.get("query", ""),
                round(float(span[0]), 3), round(float(span[1]), 3))

    def _teacher_generate_tiou(self, inputs):
        """teacher 在特权视频上独立 rollout, 返回与 inputs 等长的 per-sample r̄^T 列表。

        论文口径 (teacher_rollout_mode="sample"): 对每个 group 用【与 student 相同的
        temperature / top_p / max_new_tokens】独立采样 G 条, 取 tIoU 均值作为 r̄_g^T。
        这些 rollout 只喂给 adaptive controller, 不进 JSD loss。

        便宜版 (teacher_rollout_mode="greedy"): 每 group 贪心 1 条。方差为 0 但与
        student 的采样均值不可直接比较, 会系统性偏高(贪心通常优于采样均值) → 可能
        高估 teacher 能力。仅用于快速调试。

        与 _estimate_teacher_tiou 的区别: 后者在 student 采样出的前缀上取 argmax,
        数字是逐位生成的, 前缀一旦被 student 锚定, teacher 只能顺着 student 的答案往下补,
        因此系统性偏向 student、既高估一致性也测不出 teacher 的真实定位能力。

        跨进程一致性: 一个 group 可能横跨 2 张卡 (bpd=1 时确实如此), 各卡采样种子不同,
        若各自独立算会得到不同的 r̄_g^T → λ_g 在不同卡上不一致 → 梯度不一致。
        因此这里 gather 全部 (group_key -> sum,count) 再归并, 保证所有卡看到同一个 r̄_g^T。

        失败返回 None, 调用方自动回退。
        """
        device = self.accelerator.device
        tok = self._tokenizer()
        mod = self._get_teacher_module(self.model)

        n_roll = self.teacher_rollout_n or self.num_generations
        do_sample = (self.teacher_rollout_mode == "sample")
        if not do_sample:
            n_roll = 1

        was_training = mod.training
        prev_cache = getattr(mod.config, "use_cache", False)
        mod.eval()
        mod.config.use_cache = True

        gen_extra = {}
        try:
            import inspect as _inspect
            if "use_model_defaults" in _inspect.signature(mod.generate).parameters:
                gen_extra["use_model_defaults"] = False
        except Exception:
            pass
        if do_sample:
            gen_extra.update(
                do_sample=True,
                temperature=self.temperature,
                top_p=getattr(self.args, "top_p", 1.0) or 1.0,
                top_k=getattr(self.args, "top_k", None),
            )
        else:
            gen_extra.update(do_sample=False)

        local = {}   # group_key -> [tiou_sum, count]
        ok = True
        try:
            with torch.no_grad():
                seen = set()
                for d in inputs:
                    gkey = self._group_key(d)
                    if gkey in seen:
                        continue          # 同 group 在本卡的重复条目只算一次
                    seen.add(gkey)

                    t_ids = d.get("teacher_input_ids", None)
                    if t_ids is None:
                        local[gkey] = [0.0, 1]
                        continue
                    t_ids = t_ids.unsqueeze(0).to(device)
                    t_mask = d.get("teacher_attention_mask", None)
                    t_mask = (t_mask.unsqueeze(0).to(device) if t_mask is not None
                              else torch.ones_like(t_ids))

                    kw = {}
                    for k in QWEN_2_5_VL_KEY_WORDS:
                        tk = f"teacher_{k}"
                        if tk not in d:
                            continue
                        v = d[tk]
                        if k == "mm_token_type_ids" and isinstance(v, torch.Tensor):
                            v = v.unsqueeze(0) if v.dim() == 1 else v
                        kw[k] = v.to(device) if isinstance(v, torch.Tensor) else v

                    gt = (gkey[2], gkey[3])
                    tot, cnt = 0.0, 0
                    # 分批: n_roll 条一次性生成会把 KV cache 峰值 ×n_roll,
                    # 14k prompt token 下容易 OOM, 故按 teacher_rollout_batch 切。
                    chunk = self.teacher_rollout_batch or n_roll
                    remaining = n_roll
                    while remaining > 0:
                        nb = min(chunk, remaining)
                        out = mod.generate(
                            input_ids=t_ids, attention_mask=t_mask,
                            max_new_tokens=self.teacher_gen_max_new_tokens,
                            num_return_sequences=nb,
                            pad_token_id=tok.pad_token_id,
                            eos_token_id=tok.eos_token_id,
                            **kw, **gen_extra,
                        )
                        for r in range(out.size(0)):
                            text = tok.decode(out[r, t_ids.size(1):], skip_special_tokens=True)
                            ts = extract_time(text)
                            tot += iou(ts[-1], gt) if ts else 0.0
                            cnt += 1
                        del out
                        remaining -= nb
                    local[gkey] = [tot, cnt]
        except Exception as e:
            if not self._teacher_gen_warned:
                self._teacher_gen_warned = True
                print(f"[OPD] teacher rollout 失败, 本次回退: {type(e).__name__}: {e}", flush=True)
            ok = False
        finally:
            mod.config.use_cache = prev_cache
            if was_training:
                mod.train()
            torch.cuda.empty_cache()

        if not ok:
            return None

        # 跨进程归并: 保证同一 group 在所有卡上拿到相同的 r̄_g^T
        try:
            from accelerate.utils import gather_object
            shards = gather_object([local])
        except Exception:
            shards = [local]
        merged = {}
        for sh in shards:
            for k, (s, c) in sh.items():
                acc = merged.setdefault(k, [0.0, 0])
                acc[0] += s
                acc[1] += c
        gmean = {k: (s / c if c else 0.0) for k, (s, c) in merged.items()}
        self._last_teacher_group_tiou = gmean
        return [gmean.get(self._group_key(d), 0.0) for d in inputs]

    def log(self, logs, *args, **kwargs):
        mode = "train" if self.model.training else "eval"
        removed = {}
        for key in self._OPD_METRIC_KEYS:
            if key in self._metrics[mode] and len(self._metrics[mode][key]) == 0:
                removed[key] = self._metrics[mode].pop(key)
        super().log(logs, *args, **kwargs)
        for key in self._OPD_METRIC_KEYS:
            if key not in self._metrics[mode]:
                self._metrics[mode][key] = removed.get(key, [])

    # ------------------------------------------------------------------
    # Override: chunked forward with proper Qwen VL visual kwargs splitting
    # ------------------------------------------------------------------
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, **forward_kwargs):
        _inner = model.module if hasattr(model, 'module') else model

        # Qwen3-VL(transformers>=5): mm_token_type_ids 需与 input_ids 等长,尾部补0(completion是纯文本)
        if 'mm_token_type_ids' in forward_kwargs:
            _mm = forward_kwargs['mm_token_type_ids']
            if _mm.dim() == 1:
                _mm = _mm.unsqueeze(0)
            _pad = input_ids.size(1) - _mm.size(1)
            if _pad > 0:
                _mm = torch.cat([_mm, torch.zeros(_mm.size(0), _pad, dtype=_mm.dtype, device=_mm.device)], dim=1)
            elif _pad < 0:
                _mm = _mm[:, :input_ids.size(1)]
            forward_kwargs['mm_token_type_ids'] = _mm

        if batch_size is None or batch_size >= input_ids.size(0):
            backbone_out = _inner.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **forward_kwargs,
            )
            logits = _inner.lm_head(backbone_out[0][:, -(logits_to_keep + 1):-1, :])
            del backbone_out
            logits = logits / self.temperature
            return selective_log_softmax(logits, input_ids[:, -logits_to_keep:])

        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            end = min(i + batch_size, input_ids.size(0))
            chunk_kwargs = _split_visual_kwargs(forward_kwargs, i, end)
            backbone_out = _inner.model(
                input_ids=input_ids[i:end],
                attention_mask=attention_mask[i:end],
                **chunk_kwargs,
            )
            logits = _inner.lm_head(backbone_out[0][:, -(logits_to_keep + 1):-1, :])
            del backbone_out
            logits = logits / self.temperature
            all_logps.append(selective_log_softmax(logits, input_ids[i:end, -logits_to_keep:]))
        return torch.cat(all_logps, dim=0)

    # ------------------------------------------------------------------
    # Override: collect teacher tensors alongside student tensors
    # ------------------------------------------------------------------
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Generate completions and collect both student and teacher vision tensors."""
        teacher_collected = {}
        for key in (*TEACHER_VISION_KEYS, "teacher_input_ids", "teacher_attention_mask", "teacher_num_visuals", "teacher_mm_token_type_ids"):
            if key in inputs[0]:
                teacher_collected[key] = [d[key] for d in inputs]

        gt_spans = None
        # dead_rescue(fixed模式) 也需要估计 teacher tIoU; legacy模式复刻Run G: 不估计
        if self.teacher_quality_gate or (self.dead_rescue_enabled and self.dead_rescue_mode == "fixed"):
            gt_spans = []
            for inp in inputs:
                anno = inp.get("anno", {})
                span = anno.get("span", [[0, 0]])
                if isinstance(span[0], list):
                    span = span[0]
                gt_spans.append(span)

        result = super()._generate_and_score_completions(inputs)
        result.update(teacher_collected)

        if gt_spans is not None:
            gt_spans_t = torch.tensor(gt_spans, dtype=torch.float32)
            num_completions = result["advantages"].shape[0]
            repeat_factor = max(1, num_completions // len(inputs))
            result["gt_spans"] = gt_spans_t.repeat_interleave(repeat_factor, dim=0)

        # 死组检测(fixed): 用父类在全卡gather的rewards上算的组级统计(group_is_std_zero/group_reward_mean),
        # 已切好本卡切片, 与 completions 逐条对齐; 下游 shuffle/split 对所有key统一处理, 保持对齐。
        # 组跨卡分布(每卡仅数条completion), 本地 advantages 凑不出整组——故不能用本地张量检测(历史bug)。
        # 语义与Run G一致: 组内零方差 且 组reward均值 < dead_reward_mean_threshold。
        # legacy 模式不用此路径 (在 _compute_loss 里从 textual_logs 检测, 复刻Run G)
        if self.dead_rescue_enabled and self.dead_rescue_mode == "fixed":
            _gz, _gm = result.get("group_is_std_zero"), result.get("group_reward_mean")
            _gaz = result.get("group_all_zero")
            if self.dead_group_criterion == "all_zero" and _gaz is not None:
                # 论文口径: r_{g,i}=0 ∀i。与 std_zero_lowmean 的区别在于全组同为 0.25 时
                # std=0 但不是 all-zero, 前者会误判为 dead。
                _dm = _gaz.float()
                result["dead_mask"] = _dm
                if self.accelerator.is_main_process:
                    print(f"[DEADDBG] step={getattr(self.state,'global_step',-1)} "
                          f"criterion=all_zero 本卡dead样本={int(_dm.sum().item())}/{_dm.numel()}",
                          flush=True)
            elif _gz is not None and _gm is not None:
                _dm = (_gz & (_gm < self.dead_reward_mean_threshold)).float()
                result["dead_mask"] = _dm
                if self.accelerator.is_main_process:
                    print(f"[DEADDBG] step={getattr(self.state,'global_step',-1)} "
                          f"本卡dead样本={int(_dm.sum().item())}/{_dm.numel()}", flush=True)
            elif self.accelerator.is_main_process:
                print(f"[DEADDBG] group统计缺失: gz={_gz is not None} gm={_gm is not None}", flush=True)

        # teacher 独立 rollout 的 r̄^T: 每 rollout block 只算一次, 作为普通 tensor 返回,
        # 随父类的统一 shuffle/split 对齐到各切片(与 dead_mask 完全同一条对齐路径)。
        # teacher_tiou_mode="argmax"(默认) 且 controller=off 时不产生该键 → 走原路径, 行为不变。
        _need_teacher_roll = (
            self.teacher_tiou_mode == "generate" or self.adaptive_controller == "paper"
        )
        if _need_teacher_roll and gt_spans is not None:
            _tt = self._teacher_generate_tiou(inputs)
            if _tt is not None:
                _tt_t = torch.tensor(_tt, dtype=torch.float32)
                _rf = max(1, result["advantages"].shape[0] // len(inputs))
                result["teacher_tiou_gen"] = _tt_t.repeat_interleave(_rf, dim=0)
        # ==============================================================
        # 论文 Eq5: 组级 adaptive controller
        #   λ_g = 0   if r̄_g^T ≤ r̄_g^S            (teacher 无优势 → 完全不蒸馏)
        #       = 2   if r̄_g^T > r̄_g^S 且 该组全零  (dead group 且 teacher 有优势 → 加倍)
        #       = 1   otherwise
        # 这里在【组级】判定后展开到 completion 级, 使其能随后续 shuffle/split 正确对齐。
        # 原实现是把整个 batch 的 teacher_advantage 取 .mean() 后缩放整个 loss (batch级近似),
        # 且仅在 dead_rescue 分支内生效; 这里改为论文口径的逐组 λ。
        # ==============================================================
        if self.adaptive_controller == "paper":
            _rt = result.get("teacher_tiou_gen", None)
            _rs = result.get("group_reward_mean", None)
            _az = result.get("group_all_zero", None)
            if _rt is not None and _rs is not None:
                _rs = _rs.detach().float().cpu()
                _rt = _rt.detach().float().cpu()
                lam = torch.ones_like(_rs)
                no_edge = _rt <= _rs + self.teacher_advantage_margin
                lam[no_edge] = self.controller_lambda_no_edge          # 论文: 0.0
                if _az is not None:
                    _az = _az.detach().bool().cpu()
                    lam[(~no_edge) & _az] = self.controller_lambda_dead  # 论文: 2.0
                result["lambda_adapt"] = lam
                if self.accelerator.is_main_process:
                    _n = lam.numel()
                    print(f"[CTRL] step={getattr(self.state,'global_step',-1)} "
                          f"r̄T={_rt.mean().item():.4f} r̄S={_rs.mean().item():.4f} "
                          f"λ=0:{(lam == self.controller_lambda_no_edge).sum().item()}/{_n} "
                          f"λ=2:{(lam == self.controller_lambda_dead).sum().item()}/{_n} "
                          f"λ̄={lam.mean().item():.3f}", flush=True)
            elif self.accelerator.is_main_process and not self._ctrl_warned:
                self._ctrl_warned = True
                print(f"[CTRL] 警告: controller=paper 但缺少统计 "
                      f"(teacher_tiou_gen={_rt is not None}, group_reward_mean={_rs is not None}) "
                      f"→ 本次退化为 λ≡1", flush=True)

        return result

    # ------------------------------------------------------------------
    # Override: merge teacher tensors the same way as student tensors
    # ------------------------------------------------------------------
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Prepare inputs, including teacher vision tensor merging."""
        inputs = super()._prepare_inputs(generation_batch)

        device = inputs["prompt_ids"].device

        if "gt_spans" in inputs and isinstance(inputs["gt_spans"], torch.Tensor):
            inputs["gt_spans"] = inputs["gt_spans"].to(device)
        if "dead_mask" in inputs and isinstance(inputs["dead_mask"], torch.Tensor):
            inputs["dead_mask"] = inputs["dead_mask"].to(device)
        # teacher rollout tIoU / 逐completion reward / 组级 λ / all-zero 标记
        for _k in ("teacher_tiou_gen", "rewards", "lambda_adapt", "group_all_zero"):
            if _k in inputs and isinstance(inputs[_k], torch.Tensor):
                inputs[_k] = inputs[_k].to(device)

        for key in TEACHER_TENSOR_KEYS:
            if key in inputs and isinstance(inputs[key], list):
                inputs[key] = torch.cat(inputs[key]).to(device)
        for key in TEACHER_LIST_KEYS:
            if key in inputs and isinstance(inputs[key], list):
                if isinstance(inputs[key][0], list):
                    inputs[key] = [t for ts in inputs[key] for t in ts]

        if "teacher_input_ids" in inputs and isinstance(inputs["teacher_input_ids"], list):
            ids_list = inputs["teacher_input_ids"]
            mask_list = inputs.get("teacher_attention_mask", [torch.ones_like(t) for t in ids_list])
            pad_id = getattr(self.processing_class, "pad_token_id", 0) or 0
            max_len = max(t.size(0) for t in ids_list)
            batch_size = len(ids_list)
            padded_ids = torch.full((batch_size, max_len), pad_id, dtype=ids_list[0].dtype, device=device)
            padded_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
            for i, (t_ids, t_mask) in enumerate(zip(ids_list, mask_list)):
                L = t_ids.size(0)
                padded_ids[i, max_len - L :] = t_ids
                padded_mask[i, max_len - L :] = t_mask
            inputs["teacher_prompt_ids"] = padded_ids
            inputs["teacher_prompt_mask"] = padded_mask
            inputs.pop("teacher_input_ids", None)
            inputs.pop("teacher_attention_mask", None)

            # Qwen3-VL: teacher mm_token_type_ids 与 teacher_prompt_ids 右对齐 padding (=0)
            if "teacher_mm_token_type_ids" in inputs and isinstance(inputs["teacher_mm_token_type_ids"], list):
                mm_list = inputs["teacher_mm_token_type_ids"]
                padded_mm = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
                for i, t_mm in enumerate(mm_list):
                    L = t_mm.size(0)
                    padded_mm[i, max_len - L :] = t_mm
                inputs["teacher_mm_token_type_ids"] = padded_mm

        return inputs

    # ------------------------------------------------------------------
    # Teacher quality estimation from argmax tokens
    # ------------------------------------------------------------------
    def _estimate_teacher_tiou(self, teacher_logits, completion_ids, completion_mask, gt_spans):
        """Estimate teacher tIoU by decoding teacher argmax tokens.

        Teacher-forced argmax: at each position, take teacher's most likely
        token. Decode, parse as time range, compute tIoU vs GT.
        This is an approximation — not autoregressive generation.
        """
        teacher_preds = teacher_logits.argmax(dim=-1)  # (B, L)
        tokenizer = self.processing_class
        tious = []
        for i in range(teacher_preds.size(0)):
            mask_i = completion_mask[i].bool()
            tokens_i = teacher_preds[i][mask_i]
            text_i = tokenizer.decode(tokens_i, skip_special_tokens=True)
            timestamps = extract_time(text_i)
            if timestamps and gt_spans is not None:
                gt = (gt_spans[i][0].item(), gt_spans[i][1].item())
                tious.append(iou(timestamps[-1], gt))
            else:
                tious.append(0.0)
        return torch.tensor(tious, dtype=torch.float32, device=teacher_logits.device)

    # ------------------------------------------------------------------
    # Override: student forward + teacher forward + JSD + GRPO loss
    # ------------------------------------------------------------------
    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # --- Student forward ---
        # Call backbone directly, then lm_head only on completion tokens.
        # Full model forward computes lm_head on ALL ~15k tokens (4.5 GiB),
        # and backward recomputes the same allocation for gradients.
        student_forward_kwargs = {
            key: inputs[key] for key in inputs if key in QWEN_2_5_VL_KEY_WORDS
        }
        # Qwen3-VL: mm_token_type_ids 只覆盖 prompt 长度, 补 0 到 P+C
        if 'mm_token_type_ids' in student_forward_kwargs:
            _mm = student_forward_kwargs['mm_token_type_ids']
            if _mm.dim() == 1:
                _mm = _mm.unsqueeze(0)
            _pad = input_ids.size(1) - _mm.size(1)
            if _pad > 0:
                _mm = torch.cat([_mm, torch.zeros(_mm.size(0), _pad, dtype=_mm.dtype, device=_mm.device)], dim=1)
            elif _pad < 0:
                _mm = _mm[:, :input_ids.size(1)]
            student_forward_kwargs['mm_token_type_ids'] = _mm
        _inner = model.module if hasattr(model, 'module') else model
        _backbone_out = _inner.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **student_forward_kwargs,
        )
        student_logits = _inner.lm_head(
            _backbone_out[0][:, -(logits_to_keep + 1):-1, :]
        )
        del _backbone_out

        per_token_logps = selective_log_softmax(
            student_logits / self.temperature, input_ids[:, -logits_to_keep:]
        )

        # Free memory before teacher forward
        torch.cuda.empty_cache()

        # --- Teacher forward + JSD distillation ---
        teacher_forward_kwargs = {}
        for key in QWEN_2_5_VL_KEY_WORDS:
            teacher_key = f"teacher_{key}"
            if teacher_key in inputs:
                teacher_forward_kwargs[key] = inputs[teacher_key]

        jsd_loss = torch.tensor(0.0, device=input_ids.device)
        has_teacher = len(teacher_forward_kwargs) > 0
        teacher_top1 = None
        rlsd_tlogp_list = []   # RLSD: 收集teacher在student每token上的logp(当幅度乘子用)

        eff_kl_weight = self._current_kl_weight()
        # 论文 Eq5 controller: 若本切片所有样本的 λ_g 都是 0, teacher 前向的结果一定被乘成 0,
        # 直接跳过可省一次全量 teacher forward。实测 teacher 优势组约占一半 → 平均省下约
        # 一半的 teacher 前向, 正好补偿新增的 teacher rollout 开销。
        _lam_in = inputs.get("lambda_adapt", None) if self.adaptive_controller == "paper" else None
        _skip_by_ctrl = bool(
            _lam_in is not None and self.controller_skip_zero and float(_lam_in.abs().sum()) == 0.0
        )
        if _skip_by_ctrl:
            self._metrics["train" if self.model.training else "eval"]["ctrl_skip_teacher"].append(1.0)
        if has_teacher and (self.kl_weight_start > 0 or self.kl_weight_end > 0) and not _skip_by_ctrl:
            if "teacher_prompt_ids" in inputs:
                teacher_input_ids = torch.cat([inputs["teacher_prompt_ids"], completion_ids], dim=1)
                teacher_attention_mask = torch.cat([inputs["teacher_prompt_mask"], completion_mask], dim=1)
            else:
                teacher_input_ids = input_ids
                teacher_attention_mask = attention_mask

            # Qwen3-VL: teacher mm_token_type_ids 补 0 到 teacher seq len
            if 'mm_token_type_ids' in teacher_forward_kwargs:
                _tmm = teacher_forward_kwargs['mm_token_type_ids']
                if _tmm.dim() == 1:
                    _tmm = _tmm.unsqueeze(0)
                _tpad = teacher_input_ids.size(1) - _tmm.size(1)
                if _tpad > 0:
                    _tmm = torch.cat([
                        _tmm, torch.zeros(_tmm.size(0), _tpad, dtype=_tmm.dtype, device=_tmm.device)
                    ], dim=1)
                elif _tpad < 0:
                    _tmm = _tmm[:, :teacher_input_ids.size(1)]
                teacher_forward_kwargs['mm_token_type_ids'] = _tmm

            # --- Cached snapshot teacher ---
            # teacher_update_strategy="cached_snapshot":
            #   同一 rollout block 的 spg 次更新共享同一组 teacher logits。
            #   仅在 block 第0次更新时做一次 teacher forward, 后续3次复用。
            use_cached_snapshot = (self.teacher_update_strategy == "cached_snapshot")
            spg = getattr(self.args, "steps_per_generation", 1)
            current_step = getattr(self.state, "global_step", 0)
            block_id = current_step // spg if spg > 0 else 0
            update_idx_in_block = current_step % spg if spg > 0 else 0

            # 死组检测: 两种模式
            #   fixed:  已在 _generate_and_score_completions (shuffle前, 整组连续) 算好,
            #           dead_mask 随统一 shuffle/split 对齐到当前切片
            #   legacy: Run B/G 时代行为——从 _textual_logs 读全block rewards 检测(能触发),
            #           但下游加权是 ×平均权重 (死组无相对放大, 见 rescue 块)
            dead_mask = None
            teacher_advantage = None
            if self.dead_rescue_enabled:
                if self.dead_rescue_mode == "fixed":
                    dead_mask = inputs.get("dead_mask", None)
                    if self.accelerator.is_main_process:
                        _s = "None" if dead_mask is None else f"sum={dead_mask.sum().item():.0f}/{dead_mask.numel()}"
                        print(f"[DEADDBG-LOSS] step={getattr(self.state,'global_step',-1)} dead_mask={_s}", flush=True)
                else:  # legacy
                    _r = None
                    if hasattr(self, "_textual_logs"):
                        try:
                            all_r = self._textual_logs.get("rewards", {})
                            if isinstance(all_r, dict) and all_r:
                                first_key = list(all_r.keys())[0]
                                r_list = all_r[first_key]
                                if r_list:
                                    _r = torch.tensor(r_list, device=input_ids.device)
                        except (IndexError, KeyError):
                            pass
                    if _r is not None:
                        ng = max(1, int(getattr(self, "num_generations", 1)))
                        B = _r.size(0)
                        if B % ng == 0:
                            r_g = _r.view(B // ng, ng)
                            r_std = r_g.std(dim=-1)
                            r_mean = r_g.mean(dim=-1)
                            is_dead_group = (r_std < self.dead_reward_std_eps) & (r_mean < self.dead_reward_mean_threshold)
                            dead_mask = is_dead_group.unsqueeze(-1).expand(-1, ng).reshape(-1).float()

            # 需要重新计算 teacher logits 的条件:
            #   - 不用 cached_snapshot → 每步都重新算
            #   - 用 cached_snapshot 且是 block 第0步 或 block 变了 → 重新算
            need_recompute = True
            if use_cached_snapshot:
                if update_idx_in_block == 0 or self._cached_block_id != block_id:
                    need_recompute = True
                else:
                    need_recompute = False

            try:
                batch_size = teacher_input_ids.size(0)
                per_sample_jsd_list = []
                teacher_top1_list = []
                use_gate = self.teacher_quality_gate and "gt_spans" in inputs
                # teacher_tiou_list: use_gate 或 dead_rescue 都需要
                teacher_tiou_list = [] if (use_gate or (self.dead_rescue_enabled and self.dead_rescue_mode == "fixed")) else None
                overlap_list = [] if self.golden_keep_frac < 1.0 else None

                # ExOPD: ref = 生成时 student 自身的 log-probs (stop-grad), 无需独立模型。
                ref_logps_ex = None
                if self.divergence == "exopd":
                    ref_logps_ex = (inputs["old_per_token_logps"]
                                    if inputs.get("old_per_token_logps") is not None
                                    else per_token_logps.detach())

                # 每样本视觉块数 → 累积偏移, 支持多video/多image teacher(zoom/t2sgrid)。
                # 没有该键时退回"每样本1块"(旧行为, 单video策略)。
                _num_vis = inputs.get("teacher_num_visuals", None)
                if _num_vis is not None:
                    _vis_end = torch.cumsum(_num_vis.to(torch.long), 0).tolist()
                    _vis_start = [0] + _vis_end[:-1]
                else:
                    _vis_start = list(range(batch_size)); _vis_end = list(range(1, batch_size + 1))

                # --- Teacher logits: 重新计算 or 复用缓存 ---
                if need_recompute:
                    t_logits_list = []
                    for i in range(batch_size):
                        torch.cuda.empty_cache()
                        chunk_kwargs = _split_visual_kwargs(teacher_forward_kwargs, _vis_start[i], _vis_end[i])
                        # mm_token_type_ids 是"每样本一行"的逐token张量, 必须按 batch 下标切,
                        # 不能跟视觉块下标(_vis_start/_vis_end)走, 否则多视觉策略会错位。
                        if 'mm_token_type_ids' in teacher_forward_kwargs:
                            chunk_kwargs['mm_token_type_ids'] = teacher_forward_kwargs['mm_token_type_ids'][i:i+1]

                        with torch.no_grad():
                            # Call backbone directly, then lm_head only on completion tokens.
                            # The full model's lm_head computes logits for ALL ~15k tokens
                            # (num_logits_to_keep unsupported), causing OOM even at batch=1.
                            # frozen_teacher=None → student本体(exp1); 否则用冻结初始模型(exp4)。
                            _inner = self._get_teacher_module(model)
                            backbone_out = _inner.model(
                                input_ids=teacher_input_ids[i:i+1],
                                attention_mask=teacher_attention_mask[i:i+1],
                                **chunk_kwargs,
                            )
                            hidden = backbone_out[0]
                            t_logits_i = _inner.lm_head(
                                hidden[:, -(logits_to_keep + 1):-1, :]
                            )
                            del hidden, backbone_out

                        t_logits_list.append(t_logits_i)
                        teacher_top1_list.append(t_logits_i.argmax(dim=-1))

                    # 缓存给后续 block 内更新用
                    if use_cached_snapshot:
                        self._cached_teacher_logits = t_logits_list
                        self._cached_block_id = block_id
                        self._cached_teacher_checksum = hash(
                            tuple(t.sum().item() for t in t_logits_list)
                        )
                else:
                    # 复用缓存
                    t_logits_list = self._cached_teacher_logits
                    # 校验: 同一block内缓存不变
                    current_checksum = hash(
                        tuple(t.sum().item() for t in t_logits_list)
                    )
                    assert current_checksum == self._cached_teacher_checksum, \
                        f"cached_snapshot teacher checksum mismatch at block {block_id}, update {update_idx_in_block}"
                    for t_i in t_logits_list:
                        teacher_top1_list.append(t_i.argmax(dim=-1))

                # --- JSD distillation per sample ---
                # [修4] numeric focus mask (整批算一次, 逐样本取切片)
                focus_mask_all = None
                if self.jsd_token_focus != "off":
                    focus_mask_all = self._build_numeric_focus_mask(completion_ids, completion_mask)
                for i in range(batch_size):
                    t_logits_i = t_logits_list[i]

                    # Golden-gating: teacher-student top-k overlap 作为轨迹价值分
                    if overlap_list is not None and not use_cached_snapshot:
                        kov = 16
                        s_top = student_logits[i:i+1].topk(kov, dim=-1).indices  # (1,L,kov)
                        t_top = t_logits_i.topk(kov, dim=-1).indices
                        match = (s_top.unsqueeze(-1) == t_top.unsqueeze(-2)).any(-1).float()  # (1,L,kov)
                        ov_tok = match.sum(-1) / kov  # (1,L)
                        ov_i = (ov_tok * completion_mask[i:i+1]).sum() / completion_mask[i:i+1].sum().clamp(min=1.0)
                        overlap_list.append(ov_i.detach())

                    if self.divergence == "exopd":
                        # ExOPD: dense 隐式奖励 PG, 让 student 越过弱 teacher。
                        # R_t = λ·(logπ_T(o_t) − logπ_ref(o_t)); 损失 = -(R·logπ_S)
                        t_logp_i = selective_log_softmax(
                            t_logits_i.detach() / self.temperature, completion_ids[i:i+1]
                        )  # (1, L)
                        R_t = self.exopd_lambda * (t_logp_i - ref_logps_ex[i:i+1])  # (1, L)
                        m_i = completion_mask[i:i+1]
                        exopd_i = -(
                            (R_t.detach() * per_token_logps[i:i+1] * m_i).sum(-1)
                            / m_i.sum(-1).clamp(min=1.0)
                        )
                        per_sample_jsd_list.append(exopd_i)
                    elif self.divergence == "rlsd":
                        # RLSD: 只取teacher在student token上的logp, 不做分布匹配(不当生成目标→不泄漏)。
                        t_logp_i = selective_log_softmax(
                            t_logits_i.detach() / self.temperature, completion_ids[i:i+1]
                        )  # (1, L)
                        rlsd_tlogp_list.append(t_logp_i)
                        per_sample_jsd_list.append(torch.zeros(1, device=input_ids.device))
                    else:
                        _, jsd_i = _compute_jsd_topk(
                            student_logits[i:i+1], t_logits_i.detach(),
                            self.kl_temperature, self.jsd_alpha,
                            self.distillation_topk, completion_mask[i:i+1],
                            return_per_sample=True,
                            divergence=self.divergence, clip_max=self.jsd_clip_max,
                            entropy_gate=self.entropy_gate, entropy_tau=self.entropy_tau,
                            focus_mask=(focus_mask_all[i:i+1] if focus_mask_all is not None else None),
                            focus_mode=self.jsd_token_focus,
                            focus_weight=self.jsd_numeric_weight,
                        )
                        per_sample_jsd_list.append(jsd_i)

                    if teacher_tiou_list is not None:
                        # [修3a] 优先用 teacher 自主解码的 tIoU (每block算一次, 已随shuffle对齐);
                        # 没有时回退到原 teacher-forced argmax 估计, 保持老行为。
                        _tt_gen = inputs.get("teacher_tiou_gen", None)
                        if _tt_gen is not None:
                            teacher_tiou_list.append(_tt_gen[i:i+1].to(t_logits_i.device).float())
                        else:
                            tiou_i = self._estimate_teacher_tiou(
                                t_logits_i, completion_ids[i:i+1],
                                completion_mask[i:i+1], inputs["gt_spans"][i:i+1],
                            )
                            teacher_tiou_list.append(tiou_i)

                    del t_logits_i

                per_sample_jsd = torch.cat(per_sample_jsd_list, dim=0)
                teacher_top1 = torch.cat(teacher_top1_list, dim=0)

                # Golden-gating: 每个prompt组内按overlap保留top-k条(Prompt-Min1), 其余不蒸馏
                golden_keep = None
                if overlap_list is not None and len(overlap_list) == per_sample_jsd.size(0):
                    overlaps = torch.stack(overlap_list)  # (B,)
                    ng = max(1, int(getattr(self, "num_generations", 1)))
                    B = overlaps.size(0)
                    if B % ng == 0 and ng > 1:
                        G = B // ng
                        k_keep = max(1, math.ceil(self.golden_keep_frac * ng))  # Prompt-Min1
                        ov_g = overlaps.view(G, ng)
                        idx = ov_g.topk(k_keep, dim=-1).indices  # (G, k_keep)
                        keep_g = torch.zeros_like(ov_g)
                        keep_g.scatter_(1, idx, 1.0)
                        golden_keep = keep_g.view(-1)  # (B,)

                if use_gate:
                    teacher_tiou = torch.cat(teacher_tiou_list, dim=0)
                    jsd_gate = teacher_tiou.clamp(min=0.0)
                    if golden_keep is not None:
                        jsd_gate = jsd_gate * golden_keep
                    jsd_loss = (per_sample_jsd * jsd_gate).sum() / jsd_gate.sum().clamp(min=1e-8)
                elif golden_keep is not None:
                    # 只在保留的轨迹上平均(Prompt-Min1保证每组≥1, 分母不为0)
                    jsd_loss = (per_sample_jsd * golden_keep).sum() / golden_keep.sum().clamp(min=1e-8)
                else:
                    jsd_loss = per_sample_jsd.mean()

                # --- 记录 raw JSD (加权前) ---
                jsd_raw = jsd_loss.detach().item()

                # --- Dead-group rescue: 对死组样本放大JSD权重 ---
                dead_rescue_triggered = False
                negative_teacher_downweighted = False
                teacher_gate_applied = False
                sample_weight_mean = 1.0

                # ==========================================================
                # 论文 Eq6: L_JSD = (1/|B|) Σ_g λ_g ℓ_g^JSD
                # λ_g 已在 _generate_and_score_completions 按【组】算好并展开到
                # completion 级, 随统一 shuffle/split 对齐到本切片。
                # 这里直接逐样本乘 λ 后对样本取平均: 由于每个组在本切片内贡献的样本数
                # 相同(RepeatSampler 保证), 样本均值是 (1/|B|)Σ_g λ_g ℓ_g 的无偏估计。
                # 该分支与下面的 dead_rescue 互斥 —— controller 已经包含了 dead 放大。
                # ==========================================================
                if _lam_in is not None:
                    lam_s = _lam_in.to(per_sample_jsd.device).float().reshape(-1)
                    if lam_s.numel() != per_sample_jsd.numel():
                        _n = min(lam_s.numel(), per_sample_jsd.numel())
                        lam_s = lam_s[:_n]
                        per_sample_jsd = per_sample_jsd[:_n]
                    if golden_keep is not None:
                        lam_s = lam_s * golden_keep.to(lam_s.device).reshape(-1)[:lam_s.numel()]
                    jsd_loss = (per_sample_jsd * lam_s).mean()
                    sample_weight_mean = lam_s.mean().item()
                    dead_rescue_triggered = bool((lam_s == self.controller_lambda_dead).any().item())
                    negative_teacher_downweighted = bool(
                        (lam_s == self.controller_lambda_no_edge).any().item()
                    )
                    teacher_gate_applied = True
                elif self.dead_rescue_enabled and dead_mask is not None and not use_gate:
                    # 死组样本: reward 分布 std≈0 且 mean 低, GRPO 信号消失
                    base_weight = 1.0
                    rescue_weight = dead_mask * (self.dead_rescue_multiplier - base_weight)
                    total_weight = base_weight + rescue_weight  # dead=mult, alive=1
                    sample_weight_mean = total_weight.mean().item()
                    dead_rescue_triggered = (dead_mask.mean().item() > 0)

                    if self.dead_rescue_mode == "legacy":
                        # Run B/G 时代: 平均loss × 平均权重 (死组无相对放大, 仅整体缩放)
                        # 注意 legacy 的 dead_mask 是全block(32条)而 per_sample_jsd 是切片(8条),
                        # 不能逐样本乘——这正是当时的无效化 bug, 原样保留
                        jsd_loss = jsd_loss * total_weight.mean()
                    else:
                        # fixed: 逐样本加权 per_sample_jsd * weight
                        per_sample_jsd_weighted = per_sample_jsd * total_weight
                        if self.dead_rescue_norm == "mean":
                            # [修2] 除以样本数而非权重和 → 死组的 JSD 被真正放大 multiplier 倍。
                            # 原 "weighted" 分母是 sum(w), 当 loss batch=1 时 (j*w)/w == j,
                            # 权重被自己的分母除掉, rescue 变成恒等变换(主实验哑火根因)。
                            jsd_loss = per_sample_jsd_weighted.mean()
                        else:
                            jsd_loss = (per_sample_jsd_weighted.sum()
                                        / total_weight.sum().clamp(min=1e-8))

                    # teacher advantage: teacher argmax tiou > student rollout tiou
                    # (legacy 模式 teacher_tiou_list=None, 此分支不进, 与Run G一致)
                    if teacher_tiou_list is not None and len(teacher_tiou_list) > 0:
                        teacher_tiou_all = torch.cat(teacher_tiou_list, dim=0)
                        # [修3b] 父类现在会返回逐completion的 "rewards"(与切片逐条对齐)。
                        # 老实验里该键不存在 → rewards=None → 整段是死代码。为保持可复现,
                        # 只有显式 teacher_gate_mode="on" 时才启用。
                        rewards = (inputs.get("rewards", None)
                                   if self.teacher_gate_mode == "on" else None)
                        if rewards is not None:
                            student_tiou = rewards.to(teacher_tiou_all.device).float()
                            if student_tiou.shape != teacher_tiou_all.shape:
                                _n = min(student_tiou.numel(), teacher_tiou_all.numel())
                                student_tiou = student_tiou.reshape(-1)[:_n]
                                teacher_tiou_all = teacher_tiou_all.reshape(-1)[:_n]
                            teacher_advantage = (teacher_tiou_all - student_tiou).mean()
                            # 当 teacher 明显优于 student 时, 进一步增强 dead group 的 JSD
                            if teacher_advantage > self.teacher_advantage_margin:
                                jsd_loss = jsd_loss * self.dead_rescue_multiplier
                                dead_rescue_triggered = True
                                teacher_gate_applied = True
                            elif teacher_advantage < -self.teacher_advantage_margin:
                                # teacher 明显差于 student, 衰减 JSD
                                jsd_loss = jsd_loss * self.negative_teacher_multiplier
                                negative_teacher_downweighted = True
                                teacher_gate_applied = True

                # --- 记录诊断指标 ---
                mode_diag = "train" if self.model.training else "eval"

                # teacher_tiou_mean: 估算 teacher 的 tIoU (如果有的话)
                if teacher_tiou_list is not None and len(teacher_tiou_list) > 0:
                    tt_all = torch.cat(teacher_tiou_list, dim=0)
                    self._metrics[mode_diag]["teacher_tiou_mean"].append(
                        self.accelerator.gather(tt_all.mean().detach()).nanmean().item()
                    )
                else:
                    self._metrics[mode_diag]["teacher_tiou_mean"].append(0.0)

                # student_reward_mean: 从 textual_logs 获取
                student_reward_val = 0.0
                if hasattr(self, "_textual_logs"):
                    try:
                        all_r = self._textual_logs.get("rewards", {})
                        if isinstance(all_r, dict) and all_r:
                            first_key = list(all_r.keys())[0]
                            r_list = all_r[first_key]
                            if r_list:
                                student_reward_val = sum(r_list) / len(r_list)
                    except (IndexError, KeyError):
                        pass
                self._metrics[mode_diag]["student_reward_mean"].append(student_reward_val)

                # teacher_advantage_mean
                if teacher_advantage is not None:
                    self._metrics[mode_diag]["teacher_advantage_mean"].append(
                        self.accelerator.gather(teacher_advantage.detach()).nanmean().item()
                    )
                else:
                    self._metrics[mode_diag]["teacher_advantage_mean"].append(0.0)

                # teacher_better_frac / teacher_worse_frac
                if teacher_tiou_list is not None and len(teacher_tiou_list) > 0 and hasattr(self, "_textual_logs"):
                    try:
                        all_r = self._textual_logs.get("rewards", {})
                        if isinstance(all_r, dict) and all_r:
                            first_key = list(all_r.keys())[0]
                            r_list = all_r[first_key]
                            if r_list:
                                tt_all = torch.cat(teacher_tiou_list, dim=0)
                                student_r = torch.tensor(r_list, device=tt_all.device)
                                # shape对齐: tt_all是per-prompt(B), student_r是per-completion(B*G)
                                # -> student_r 按组取mean转成per-prompt再比较
                                _B = tt_all.numel()
                                _G = max(1, student_r.numel() // _B)
                                if student_r.numel() != _B:
                                    _n = _B * _G
                                    student_r = student_r[:_n].reshape(_B, _G).mean(dim=1)
                                better = (tt_all > student_r + self.teacher_advantage_margin).float().mean().item()
                                worse = (tt_all < student_r - self.teacher_advantage_margin).float().mean().item()
                                self._metrics[mode_diag]["teacher_better_frac"].append(better)
                                self._metrics[mode_diag]["teacher_worse_frac"].append(worse)
                            else:
                                self._metrics[mode_diag]["teacher_better_frac"].append(0.0)
                                self._metrics[mode_diag]["teacher_worse_frac"].append(0.0)
                    except (IndexError, KeyError):
                        self._metrics[mode_diag]["teacher_better_frac"].append(0.0)
                        self._metrics[mode_diag]["teacher_worse_frac"].append(0.0)
                else:
                    self._metrics[mode_diag]["teacher_better_frac"].append(0.0)
                    self._metrics[mode_diag]["teacher_worse_frac"].append(0.0)

                # JSD raw / weighted
                self._metrics[mode_diag]["jsd_total_raw"].append(jsd_raw)
                self._metrics[mode_diag]["jsd_total_weighted"].append(jsd_loss.detach().item())
                self._metrics[mode_diag]["jsd_start_raw"].append(jsd_raw if update_idx_in_block == 0 else 0.0)
                self._metrics[mode_diag]["jsd_end_raw"].append(jsd_raw if update_idx_in_block == spg - 1 else 0.0)
                self._metrics[mode_diag]["sample_jsd_weight"].append(sample_weight_mean)

                # dead_rescue 诊断
                self._metrics[mode_diag]["dead_rescue_frac"].append(
                    dead_mask.mean().item() if dead_mask is not None else 0.0
                )
                self._metrics[mode_diag]["dead_rescue_triggered"].append(
                    1.0 if dead_rescue_triggered else 0.0
                )
                self._metrics[mode_diag]["negative_teacher_downweighted"].append(
                    1.0 if negative_teacher_downweighted else 0.0
                )
                self._metrics[mode_diag]["teacher_gate_applied"].append(
                    1.0 if teacher_gate_applied else 0.0
                )
                # [新] teacher 自主解码 tIoU / numeric focus 生效度
                _ttg = inputs.get("teacher_tiou_gen", None)
                self._metrics[mode_diag]["teacher_tiou_gen_mean"].append(
                    float(_ttg.float().mean().item()) if _ttg is not None else 0.0
                )
                if focus_mask_all is not None:
                    _ff = (focus_mask_all.sum() / completion_mask.sum().clamp(min=1.0))
                    self._metrics[mode_diag]["jsd_focus_frac"].append(float(_ff.item()))
                else:
                    self._metrics[mode_diag]["jsd_focus_frac"].append(1.0)

                # [新] 论文 Eq5 controller 诊断
                if _lam_in is not None:
                    _l = _lam_in.detach().float().reshape(-1)
                    self._metrics[mode_diag]["ctrl_lambda_mean"].append(float(_l.mean().item()))
                    self._metrics[mode_diag]["ctrl_lambda_zero_frac"].append(
                        float((_l == self.controller_lambda_no_edge).float().mean().item())
                    )
                    self._metrics[mode_diag]["ctrl_lambda_dead_frac"].append(
                        float((_l == self.controller_lambda_dead).float().mean().item())
                    )
                if _ttg is not None:
                    self._metrics[mode_diag]["teacher_rollout_tiou"].append(
                        float(_ttg.float().mean().item())
                    )

                # block / update 诊断
                self._metrics[mode_diag]["update_idx_in_block"].append(float(update_idx_in_block))
                self._metrics[mode_diag]["generation_block_id"].append(float(block_id))
                self._metrics[mode_diag]["teacher_cache_reused"].append(
                    1.0 if (use_cached_snapshot and not need_recompute) else 0.0
                )

                # 逐样本诊断 JSONL
                if self._diag_jsonl_path and self.accelerator.is_main_process:
                    import json as _json
                    with open(self._diag_jsonl_path, "a") as _f:
                        for i in range(per_sample_jsd.size(0)):
                            _is_dead = bool(dead_mask[i].item()) if (
                                dead_mask is not None and i < dead_mask.numel()) else False
                            _row = {
                                "step": self.state.global_step,
                                "block_id": block_id,
                                "update_idx": update_idx_in_block,
                                "sample_idx": i,
                                "per_sample_jsd": per_sample_jsd[i].item(),
                                # 逐样本真实权重(原来写的是整批均值, B>1 时不可用)
                                "jsd_weight": (self.dead_rescue_multiplier if _is_dead else 1.0)
                                              if dead_mask is not None else 1.0,
                                "jsd_weight_batch_mean": sample_weight_mean,
                                "is_dead": _is_dead,
                                "lambda_adapt": (float(_lam_in.reshape(-1)[i].item())
                                                 if (_lam_in is not None
                                                     and i < _lam_in.numel()) else None),
                                "teacher_rollout_tiou": (float(_ttg.reshape(-1)[i].item())
                                                         if (_ttg is not None
                                                             and i < _ttg.numel()) else None),
                                "dead_rescue_norm": self.dead_rescue_norm,
                                "teacher_gate_applied": teacher_gate_applied,
                                "teacher_cache_reused": bool(use_cached_snapshot and not need_recompute),
                            }
                            _f.write(_json.dumps(_row) + "\n")

            except ValueError as e:
                if "do not match" in str(e):
                    import logging
                    logging.warning(f"[OPD] Teacher token mismatch, skipping JSD: {e}")
                    has_teacher = False
                else:
                    raise

        # --- GRPO loss (standard clipped surrogate) ---
        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach()
            if inputs["old_per_token_logps"] is None
            else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        # RLSD: reward(advantages)定每token更新【方向】, teacher/student比值定【幅度】。
        # teacher只当credit权重, 不当生成目标 → 结构性不泄漏。jsd_loss此时=0。
        rlsd_mult = None
        if self.divergence == "rlsd" and len(rlsd_tlogp_list) > 0:
            teacher_logp = torch.cat(rlsd_tlogp_list, dim=0)            # (B, L)
            ratio = torch.exp((teacher_logp - per_token_logps.detach()).clamp(-10, 10))  # P_T/P_S
            sgn = torch.sign(advantages).unsqueeze(1)                   # (B,1) 方向
            clip = float(getattr(self, "rlsd_clip", 0.5))
            lam = float(getattr(self, "rlsd_lambda", 1.0))
            m = torch.clamp(torch.pow(ratio, sgn), 1.0 - clip, 1.0 + clip)
            rlsd_mult = (1.0 - lam) + lam * m                           # λ插值(λ=1=全RLSD)

        adv_tok = advantages.unsqueeze(1)
        if rlsd_mult is not None:
            adv_tok = adv_tok * rlsd_mult
        per_token_loss1 = coef_1 * adv_tok
        per_token_loss2 = coef_2 * adv_tok
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # two-stage: 探索期(前 frac 步)不惩罚负 advantage, 保多样性防早坍缩(REOPOLD)
        if self.two_stage_explore_frac > 0.0:
            _ms = getattr(self.state, "max_steps", 0) or getattr(self.args, "max_steps", 0)
            if _ms and (self.state.global_step / _ms) < self.two_stage_explore_frac:
                _keep = (advantages.unsqueeze(1) >= 0).float()
                per_token_loss = per_token_loss * _keep

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl_ref = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl_ref

        if self.loss_type == "grpo":
            grpo_loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            grpo_loss = (
                (per_token_loss * completion_mask).sum()
                / completion_mask.sum().clamp(min=1.0)
            )
        elif self.loss_type == "dr_grpo":
            grpo_loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # --- Combined loss ---
        loss = self.grpo_weight * grpo_loss + eff_kl_weight * jsd_loss

        # --- Logging ---
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl_ref = (per_token_kl_ref * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl_ref).nanmean().item()
            )

        self._metrics[mode]["jsd_distill"].append(
            self.accelerator.gather(jsd_loss.detach()).nanmean().item()
        )
        self._metrics[mode]["grpo_loss"].append(
            self.accelerator.gather(grpo_loss.detach()).nanmean().item()
        )
        self._metrics[mode]["weighted_grpo"].append(
            self.accelerator.gather((self.grpo_weight * grpo_loss).detach()).nanmean().item()
        )
        self._metrics[mode]["weighted_jsd"].append(
            self.accelerator.gather((eff_kl_weight * jsd_loss).detach()).nanmean().item()
        )

        # --- 死区/信号强度诊断 (轴4/5: 评估距离塑形/蒸馏是否救活死区) ---
        # adv_dead_frac: advantage≈0 的完成占比 = GRPO零梯度死区比例 (越低越好)
        # adv_abs_mean / adv_std: RL 信号强度 (越高=梯度越有料)
        with torch.no_grad():
            adv_flat = advantages.detach().reshape(-1).float()
            dead_frac = (adv_flat.abs() < 1e-6).float().mean()
            self._metrics[mode]["adv_dead_frac"].append(
                self.accelerator.gather(dead_frac).nanmean().item()
            )
            self._metrics[mode]["adv_abs_mean"].append(
                self.accelerator.gather(adv_flat.abs().mean()).nanmean().item()
            )
            if adv_flat.numel() > 1:
                self._metrics[mode]["adv_std"].append(
                    self.accelerator.gather(adv_flat.std()).nanmean().item()
                )

        if self.teacher_quality_gate and "gt_spans" in inputs and has_teacher:
            self._metrics[mode]["teacher_tiou"].append(
                self.accelerator.gather(teacher_tiou.mean().detach()).nanmean().item()
            )
            self._metrics[mode]["jsd_gate_mean"].append(
                self.accelerator.gather(jsd_gate.mean().detach()).nanmean().item()
            )

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio/region_mean"].append(
            self.accelerator.gather(clip_ratio).nanmean().item()
        )

        # Teacher/student top-1 agreement
        if has_teacher and self.kl_weight > 0 and teacher_top1 is not None:
            s_top1 = student_logits.argmax(dim=-1)
            t_top1 = teacher_top1
            agree = ((s_top1 == t_top1).float() * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["teacher_student_top1_agree"].append(
                self.accelerator.gather(agree).nanmean().item()
            )
        else:
            self._metrics[mode]["teacher_student_top1_agree"].append(1.0)

        # Periodic sample output (rank 0 only, every logging_steps)
        if (
            self.accelerator.is_main_process
            and hasattr(self, "state")
            and self.state.global_step > 0
            and self.state.global_step % self.args.logging_steps == 0
        ):
            try:
                tokenizer = self.processing_class
                n_show = min(2, completion_ids.size(0))
                print(f"\n{'='*60}")
                print(f"[Step {self.state.global_step}] OPD Training Samples")
                print(f"  grpo_loss={grpo_loss.item():.4f}  jsd_loss={jsd_loss.item():.4f}  "
                      f"total={loss.item():.4f}  grpo_weight={self.grpo_weight}  kl_weight={self.kl_weight}"
                      + (f"  gate_mean={jsd_gate.mean().item():.3f}" if self.teacher_quality_gate and "gt_spans" in inputs and has_teacher else ""))
                print(f"{'='*60}")
                for i in range(n_show):
                    mask_i = completion_mask[i].bool()
                    tokens_i = completion_ids[i][mask_i]
                    text_i = tokenizer.decode(tokens_i, skip_special_tokens=True)
                    adv_i = advantages[i].item() if advantages.dim() == 1 else advantages[i].mean().item()
                    print(f"  [{i}] completion: {text_i[:200]}")
                    print(f"      advantage={adv_i:.4f}")
                    if has_teacher and self.kl_weight > 0 and teacher_top1 is not None:
                        s_t1 = s_top1[i][mask_i]
                        t_t1 = t_top1[i][mask_i]
                        agree_i = (s_t1 == t_t1).float().mean().item()
                        print(f"      teacher-student top1 agree={agree_i:.3f}")
                print(f"{'='*60}\n")
            except Exception:
                pass

        # --- 数值合法性断言: NaN/Inf 检测 ---
        assert torch.isfinite(loss), f"[FATAL] total_loss is NaN/Inf at step {self.state.global_step}"
        assert torch.isfinite(grpo_loss), f"[FATAL] grpo_loss is NaN/Inf at step {self.state.global_step}"
        assert torch.isfinite(jsd_loss), f"[FATAL] jsd_loss is NaN/Inf at step {self.state.global_step}"

        return loss
