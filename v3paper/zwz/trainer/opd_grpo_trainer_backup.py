"""OPD-GRPO Trainer: GRPO with On-Policy Distillation for TVG.

Extends TimeLens's QwenvlGRPOTrainer to add teacher-student distillation.
The teacher sees GT-marked video, the student sees clean video.
Same model weights, different visual inputs.

Distillation uses Generalized JSD with top-k logit filtering,
following Vision-OPD's compute_self_distillation_loss.
"""

import os
import sys
from typing import Any, Union

import torch
import torch.nn.functional as F
from trl.trainer.utils import selective_log_softmax

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
)
TEACHER_LIST_KEYS = ("teacher_second_per_grid_ts",)


def _compute_jsd_topk(student_logits, teacher_logits, temperature, alpha, topk, completion_mask):
    """Compute Generalized JSD with top-k logit filtering.

    Following Vision-OPD: select top-k from teacher, add tail bucket,
    compute JSD on the (k+1)-dim distribution.

    Args:
        student_logits: (B, L, V) raw logits from student
        teacher_logits: (B, L, V) raw logits from teacher (detached)
        temperature: softmax temperature
        alpha: JSD interpolation (0.5 = symmetric)
        topk: number of top logits to keep (None = full vocabulary)
        completion_mask: (B, L) mask for valid tokens

    Returns:
        scalar JSD loss (masked mean over tokens)
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

    # Generalized JSD
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

    # Masked mean
    jsd_loss = (
        (per_token_jsd * completion_mask).sum(-1)
        / completion_mask.sum(-1).clamp(min=1.0)
    ).mean()
    return jsd_loss


class OPDGRPOTrainer(QwenvlGRPOTrainer):
    """GRPO trainer with On-Policy Distillation.

    Loss = L_grpo + kl_weight * L_jsd

    The teacher is the SAME model but receives GT-marked video input.
    Teacher forward is done with torch.no_grad().
    Distillation uses Generalized JSD with top-k (Vision-OPD style).
    """

    def __init__(
        self,
        *args,
        kl_weight: float = 1.0,
        kl_temperature: float = 1.0,
        jsd_alpha: float = 0.5,
        distillation_topk: int = 100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight
        self.kl_temperature = kl_temperature
        self.jsd_alpha = jsd_alpha
        self.distillation_topk = distillation_topk
        self._metrics["train"]["jsd_distill"] = []
        self._metrics["eval"]["jsd_distill"] = []
        self._metrics["train"]["grpo_loss"] = []
        self._metrics["eval"]["grpo_loss"] = []
        self._metrics["train"]["weighted_jsd"] = []
        self._metrics["eval"]["weighted_jsd"] = []
        self._metrics["train"]["teacher_student_top1_agree"] = []
        self._metrics["eval"]["teacher_student_top1_agree"] = []

    _OPD_METRIC_KEYS = ("jsd_distill", "grpo_loss", "weighted_jsd", "teacher_student_top1_agree")

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
    # Override: collect teacher tensors alongside student tensors
    # ------------------------------------------------------------------
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """Generate completions and collect both student and teacher vision tensors."""
        teacher_collected = {}
        for key in (*TEACHER_VISION_KEYS, "teacher_input_ids", "teacher_attention_mask"):
            if key in inputs[0]:
                teacher_collected[key] = [d[key] for d in inputs]

        result = super()._generate_and_score_completions(inputs)
        result.update(teacher_collected)
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

        return inputs

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
        student_forward_kwargs = {
            key: inputs[key] for key in inputs if key in QWEN_2_5_VL_KEY_WORDS
        }
        student_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **student_forward_kwargs,
        ).logits[:, -(logits_to_keep + 1) : -1, :]

        student_logits_scaled = student_logits / self.temperature
        per_token_logps = selective_log_softmax(
            student_logits_scaled, input_ids[:, -logits_to_keep:]
        )

        # --- Teacher forward + JSD distillation ---
        teacher_forward_kwargs = {}
        for key in QWEN_2_5_VL_KEY_WORDS:
            teacher_key = f"teacher_{key}"
            if teacher_key in inputs:
                teacher_forward_kwargs[key] = inputs[teacher_key]

        jsd_loss = torch.tensor(0.0, device=input_ids.device)
        has_teacher = len(teacher_forward_kwargs) > 0

        if has_teacher and self.kl_weight > 0:
            if "teacher_prompt_ids" in inputs:
                teacher_input_ids = torch.cat([inputs["teacher_prompt_ids"], completion_ids], dim=1)
                teacher_attention_mask = torch.cat([inputs["teacher_prompt_mask"], completion_mask], dim=1)
            else:
                teacher_input_ids = input_ids
                teacher_attention_mask = attention_mask

            try:
                with torch.no_grad():
                    teacher_logits = model(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attention_mask,
                        **teacher_forward_kwargs,
                    ).logits[:, -(logits_to_keep + 1) : -1, :]

                jsd_loss = _compute_jsd_topk(
                    student_logits, teacher_logits.detach(),
                    self.kl_temperature, self.jsd_alpha,
                    self.distillation_topk, completion_mask,
                )
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

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

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
        loss = grpo_loss + self.kl_weight * jsd_loss

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
        self._metrics[mode]["weighted_jsd"].append(
            self.accelerator.gather((self.kl_weight * jsd_loss).detach()).nanmean().item()
        )

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio/region_mean"].append(
            self.accelerator.gather(clip_ratio).nanmean().item()
        )

        # Teacher/student top-1 agreement
        if has_teacher and self.kl_weight > 0:
            s_top1 = student_logits.argmax(dim=-1)
            t_top1 = teacher_logits.argmax(dim=-1)
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
                      f"total={loss.item():.4f}  kl_weight={self.kl_weight}")
                print(f"{'='*60}")
                for i in range(n_show):
                    mask_i = completion_mask[i].bool()
                    tokens_i = completion_ids[i][mask_i]
                    text_i = tokenizer.decode(tokens_i, skip_special_tokens=True)
                    adv_i = advantages[i].item() if advantages.dim() == 1 else advantages[i].mean().item()
                    print(f"  [{i}] completion: {text_i[:200]}")
                    print(f"      advantage={adv_i:.4f}")
                    if has_teacher and self.kl_weight > 0:
                        s_t1 = s_top1[i][mask_i]
                        t_t1 = t_top1[i][mask_i]
                        agree_i = (s_t1 == t_t1).float().mean().item()
                        print(f"      teacher-student top1 agree={agree_i:.3f}")
                print(f"{'='*60}\n")
            except Exception:
                pass

        return loss
