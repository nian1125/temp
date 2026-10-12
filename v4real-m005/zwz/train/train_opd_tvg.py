"""OPD-GRPO training for Temporal Video Grounding.

Combines GRPO (temporal IoU reward) with teacher-student JSD distillation.
Teacher sees GT-marked video, student sees clean video — same model weights.

Usage:
    deepspeed zwz/train/train_opd_tvg.py \
        --anno_path /path/to/charades-timelens.json \
        --video_dir /path/to/videos/ \
        --model_name_or_path /path/to/Qwen3-VL-8B-Instruct \
        --kl_weight 1.0 \
        ...

Requires TIMELENS_ROOT env var pointing to TimeLens repo root.
"""

import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
TIMELENS_ROOT = os.environ.get("TIMELENS_ROOT", "")

if TIMELENS_ROOT and TIMELENS_ROOT not in sys.path:
    sys.path.insert(0, TIMELENS_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NPU (昇腾) 兼容: 映射 torch.cuda → torch.npu
from zwz.npu_patch import patch_npu
patch_npu()
import torch
ON_NPU = hasattr(torch, "npu") and torch.cuda.is_available() and not torch.cuda.device_count == 0
if ON_NPU:
    os.environ.setdefault("USE_FLASH_ATTENTION", "0")
    print(f"[train_opd_tvg] NPU mode detected, forcing sdpa attention")
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    HfArgumentParser,
)

from training.params import DataArguments, GRPOArguments, ModelArguments
from training.train.train_utils import (
    print_trainable_parameters,
    safe_save_model_for_hf_trainer,
)

from zwz.data.tvg_opd_dataset import TVGOPDDataset
from zwz.train.reward_funcs import load_reward_funcs
from zwz.trainer.opd_grpo_trainer import OPDGRPOTrainer


@dataclass
class OPDArguments:
    anno_path: str = field(
        default="", metadata={"help": "Path to annotation file (TimeLens JSON or JSONL)"},
    )
    video_dir: str = field(
        default="", metadata={"help": "Directory containing video files"},
    )
    dataset_name: str = field(
        default="timelens", metadata={"help": "Dataset format: timelens, charades, anet"},
    )
    kl_weight: float = field(
        default=1.0, metadata={"help": "Weight for KL distillation loss (0 = pure GRPO)"},
    )
    kl_temperature: float = field(
        default=1.0, metadata={"help": "Temperature for KL distillation softmax"},
    )
    teacher_strategy: str = field(
        default="numberit+red_border",
        metadata={"help": "Teacher marking strategy: red_border, numberit+red_border"},
    )
    teacher_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Cache directory for teacher marked videos"},
    )
    processor_path: Optional[str] = field(
        default=None,
        metadata={"help": "若设置, processor/tokenizer 从此路径加载(权重仍从model_name_or_path)。"
                          "用于 Qwen2.5-VL 借 TimeLens-7B 的 processor 拿 timestamp interleave 时间编码。"},
    )
    jsd_alpha: float = field(
        default=0.5, metadata={"help": "JSD interpolation alpha (0.5 = symmetric)"},
    )
    distillation_topk: int = field(
        default=100, metadata={"help": "Top-k logits for JSD distillation (Vision-OPD style)"},
    )
    teacher_quality_gate: bool = field(
        default=False,
        metadata={"help": "Gate JSD by teacher argmax tIoU: only distill when teacher is confident/correct"},
    )
    grpo_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for GRPO loss (increase to amplify RL signal vs JSD distillation)"},
    )
    kl_weight_end: Optional[float] = field(
        default=None,
        metadata={"help": "End value of kl_weight for schedule. None=constant (use kl_weight)."},
    )
    kl_schedule: str = field(
        default="constant",
        metadata={"help": "kl_weight schedule: constant | linear (kl_weight->kl_weight_end over training)"},
    )
    divergence: str = field(
        default="jsd",
        metadata={"help": "蒸馏散度: jsd | fkl(Forward KL=KL(teacher||student)) | rkl(Reverse KL)"},
    )
    jsd_clip_max: float = field(
        default=0.0,
        metadata={"help": "每token散度截断上限(>0启用), 防重尾梯度爆炸。0=不截断"},
    )
    exopd_lambda: float = field(
        default=1.25,
        metadata={"help": "ExOPD奖励外推因子λ(divergence=exopd时生效), 1.25为论文甜点, ≥1.5易hack"},
    )
    entropy_gate: str = field(
        default="off",
        metadata={"help": "蒸馏token加权: off | soft(按teacher熵归一化加权,聚焦边界token) | hard(只在高熵token蒸馏)"},
    )
    entropy_tau: float = field(
        default=0.8,
        metadata={"help": "entropy_gate=hard 的熵阈值"},
    )
    golden_keep_frac: float = field(
        default=1.0,
        metadata={"help": "golden-gating: 每prompt组按overlap保留的轨迹比例(<1启用,Prompt-Min1保底), 1.0=全保留"},
    )
    two_stage_explore_frac: float = field(
        default=0.0,
        metadata={"help": "two-stage: 前该比例步数为探索期, 不惩罚负advantage(REOPOLD防早坍缩), 0=关闭"},
    )
    exopd_ref_path: Optional[str] = field(
        default=None,
        metadata={"help": "ExOPD reference模型路径(4-bit加载), 默认用policy的base路径"},
    )
    student_mark_strategy: Optional[str] = field(
        default=None,
        metadata={"help": "ZwZ式: 给student输入也加标记(numberit直喂=实验B / coarse_box粗框=B1)。配 kl_weight=0 纯RL"},
    )
    zoom_expand: float = field(
        default=2.0,
        metadata={"help": "coarse_box zoom ratio (1.0=exact GT, 2.0=2x GT window)"},
    )
    freeze_teacher: bool = field(
        default=False,
        metadata={"help": "冻结teacher(exp4): 额外加载一份初始SFT模型作teacher, 全程不更新, "
                          "只训student。默认False=self-distillation(teacher=student本体, 权重随训练更新, exp1)"},
    )
    teacher_update_strategy: str = field(
        default="drifting",
        metadata={"help": "teacher更新策略: drifting(每步用当前权重, 默认) | "
                          "cached_snapshot(每rollout block快照一次, spg次更新共享)"},
    )
    dead_rescue_enabled: bool = field(
        default=False,
        metadata={"help": "Dead-group rescue: reward分布死组(std≈0且mean低)时增强JSD"},
    )
    dead_rescue_mode: str = field(
        default="fixed",
        metadata={"help": "fixed(shuffle前整组检测+逐样本加权, 默认) | "
                          "legacy(复刻Run B/G: textual_logs检测能触发, 但×平均权重死组无相对放大)"},
    )
    dead_reward_std_eps: float = field(
        default=1e-6,
        metadata={"help": "dead-group判定: reward std阈值"},
    )
    dead_reward_mean_threshold: float = field(
        default=0.30,
        metadata={"help": "dead-group判定: reward mean阈值(低于此视为死组)"},
    )
    teacher_advantage_margin: float = field(
        default=0.05,
        metadata={"help": "teacher优势阈值(teacher tiou - student tiou 超过此值时增强rescue)"},
    )
    dead_rescue_multiplier: float = field(
        default=2.0,
        metadata={"help": "死组样本的JSD权重倍数"},
    )
    negative_teacher_multiplier: float = field(
        default=0.25,
        metadata={"help": "teacher明显差于student时的JSD权重衰减倍数"},
    )
    ref_anno_path: Optional[str] = field(
        default=None,
        metadata={"help": "参考标注路径(如 TimeLens-100K), 用于构建 coarse_box_multi 的蓝框事件映射。"
                          "None=用训练数据自身构建"},
    )
    teacher_prompt_file: Optional[str] = field(
        default=None,
        metadata={"help": "teacher prompt 模板文件路径(含 {query} 占位符)。覆盖 TEACHER_PROMPT_PREFIX + GROUNDING_PROMPT。"},
    )
    ema_decay: float = field(
        default=0.0,
        metadata={"help": "EMA teacher 衰减系数(0=禁用, >0=启用 EMA teacher)。如 0.995。"},
    )
    # ================= 新增开关: 默认值均复刻原行为, 老实验可复现 =================
    teacher_tiou_mode: str = field(
        default="argmax",
        metadata={"help": "controller 的 r̄^T 来源: "
                          "forced(v4-real推荐: v1式teacher-forced argmax, 在student已采样的"
                          "completion上只做一次no_grad前向, 无生成、各卡对称、不hang) | "
                          "argmax(v1原样, 在_compute_loss里顺带算, 不供controller用) | "
                          "generate(teacher独立rollout, 语义最干净但8卡上会watchdog超时)"},
    )
    teacher_gen_max_new_tokens: int = field(
        default=48,
        metadata={"help": "teacher_tiou_mode=generate 时的最大生成token数(答案模板约18token, 48足够)"},
    )
    teacher_gate_mode: str = field(
        default="off",
        metadata={"help": "teacher 质量门控: off(默认, 复刻原行为——该分支因父类不返回rewards而是死代码) | "
                          "on(启用 teacher_advantage 门控: teacher优则放大JSD, 劣则×negative_teacher_multiplier)"},
    )
    dead_rescue_norm: str = field(
        default="weighted",
        metadata={"help": "dead-group rescue 归一化: weighted(原行为 sum(j*w)/sum(w), "
                          "loss batch=1 时是恒等变换→rescue无效) | mean(sum(j*w)/B, 死组真正放大)"},
    )
    jsd_token_focus: str = field(
        default="off",
        metadata={"help": "JSD token聚焦(解决模板token稀释答案token): off(原行为,全token均摊) | "
                          "numeric(硬聚焦, 只在数值token上蒸馏并只在其上取平均) | "
                          "numeric_soft(软加权, 数值token权重=jsd_numeric_weight, 每样本归一化)"},
    )
    jsd_numeric_weight: float = field(
        default=4.0,
        metadata={"help": "jsd_token_focus=numeric_soft 时数值token的相对权重"},
    )
    # ---- 论文 Eq5/Eq6: 组级 adaptive controller ----
    adaptive_controller: str = field(
        default="off",
        metadata={"help": "off(原行为: batch级teacher_advantage标量缩放, 仅dead_rescue内生效) | "
                          "paper(论文Eq5: 逐组 λ_g∈{no_edge,1,dead}, r̄T≤r̄S→0, "
                          "r̄T>r̄S且全零组→2, 否则1; 归一化按Eq6 除以组数)"},
    )
    controller_lambda_no_edge: float = field(
        default=0.0,
        metadata={"help": "λ_g when r̄_g^T ≤ r̄_g^S (论文=0, 即teacher无优势时完全不蒸馏)"},
    )
    controller_lambda_dead: float = field(
        default=2.0,
        metadata={"help": "λ_g when teacher有优势且组内全零 (论文=2)"},
    )
    controller_skip_zero: bool = field(
        default=True,
        metadata={"help": "λ全0的切片跳过teacher forward(纯省算力, 数学结果不变)"},
    )
    dead_group_criterion: str = field(
        default="std_zero_lowmean",
        metadata={"help": "dead group 判据: std_zero_lowmean(原行为: std≈0 且 mean<阈值) | "
                          "all_zero(论文: 组内所有rollout的reward都为0)"},
    )
    # ---- teacher 独立 rollout (controller 的 r̄^T 来源) ----
    teacher_rollout_mode: str = field(
        default="sample",
        metadata={"help": "sample(论文: 与student同temperature采样G条取均值) | "
                          "greedy(贪心1条, 便宜但会高估teacher, 仅调试用)"},
    )
    teacher_rollout_n: int = field(
        default=0,
        metadata={"help": "teacher rollout 条数, 0=用 num_generations(=G, 论文口径)"},
    )
    teacher_rollout_batch: int = field(
        default=0,
        metadata={"help": "teacher rollout 分批大小(省显存), 0=一次全量生成"},
    )
    blue_zoom_expand: float = field(
        default=-1.0,
        metadata={"help": "蓝框(干扰事件)窗口扩张系数。-1=沿用zoom_expand(历史行为); "
                          "1.0=论文口径(干扰窗口保留verified边界, 只扩红框)"},
    )


local_rank = None


def rank0_print(*args):
    if local_rank in (0, -1, None):
        print(*args)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, GRPOArguments, OPDArguments)
    )
    model_args, data_args, training_args, opd_args = (
        parser.parse_args_into_dataclasses()
    )

    local_rank = training_args.local_rank

    # Patch data_args with OPD-specific fields expected by TVGOPDDataset
    data_args.anno_path = opd_args.anno_path
    data_args.video_dir = opd_args.video_dir
    data_args.dataset_name = opd_args.dataset_name
    data_args.zoom_expand = opd_args.zoom_expand
    data_args.blue_zoom_expand = opd_args.blue_zoom_expand

    compute_dtype = (
        torch.bfloat16
        if training_args.bf16
        else (torch.float16 if training_args.fp16 else torch.float32)
    )

    # --- Load model ---
    rank0_print(f"Loading model: {model_args.model_name_or_path}")
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        attn_implementation=(
            "sdpa" if (ON_NPU or training_args.disable_flash_attn2)
            else "flash_attention_2"
        ),
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # --- Configure trainable parameters ---
    # 兼容 Qwen2.5-VL(model.visual) 和 Qwen3-VL(可能嵌套)
    _visual = None
    for attr in ('visual', 'vision_model', 'vision_tower'):
        if hasattr(model, attr):
            _visual = getattr(model, attr)
            break
    if _visual is None and hasattr(model, 'model'):
        for attr in ('visual', 'vision_model', 'vision_tower'):
            if hasattr(model.model, attr):
                _visual = getattr(model.model, attr)
                break
    if _visual is None:
        raise AttributeError(
            f"Cannot find vision tower in {type(model).__name__}. "
            f"Children: {list(dict(model.named_children()).keys())}"
        )
    for p in _visual.parameters():
        p.requires_grad = not training_args.freeze_vision_tower
    for p in _visual.merger.parameters():
        p.requires_grad = not training_args.freeze_merger
    for p in model.model.parameters():
        p.requires_grad = not training_args.freeze_llm
    for p in model.lm_head.parameters():
        p.requires_grad = not training_args.freeze_llm

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if local_rank in (0, -1):
        print_trainable_parameters(model, training_args=training_args)

    # --- Frozen teacher (exp4) ---
    # 额外加载一份初始 SFT 模型作 teacher, 全程冻结不更新, 只用于特权视频前向。
    # 默认 None = self-distillation (teacher=student本体, 权重随训练更新, exp1)。
    frozen_teacher = None
    if opd_args.freeze_teacher:
        rank0_print(f"Loading FROZEN teacher (exp4): {model_args.model_name_or_path}")
        frozen_teacher = AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation=(
                "sdpa" if (ON_NPU or training_args.disable_flash_attn2)
                else "flash_attention_2"
            ),
            trust_remote_code=True,
        )
        frozen_teacher.config.use_cache = False
        frozen_teacher.eval()
        for p in frozen_teacher.parameters():
            p.requires_grad = False

    # --- Load processor ---
    # processor_path 优先: Qwen2.5-VL权重 + TimeLens-7B processor(timestamp interleave时间编码)
    _proc_src = opd_args.processor_path or model_args.model_name_or_path
    rank0_print(f"Loading processor: {_proc_src}")
    processor = AutoProcessor.from_pretrained(
        _proc_src,
        padding_side="left",
        trust_remote_code=True,
    )
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.bos_token_id = processor.tokenizer.bos_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    processor.pad_token = processor.tokenizer.pad_token

    # --- Reward functions ---
    reward_funcs_list = load_reward_funcs(
        [n.strip() for n in training_args.reward_funcs.split(",")]
    )

    # --- Teacher prompt override: 从文件读取 ---
    _teacher_prompt_override = None
    if opd_args.teacher_prompt_file:
        with open(opd_args.teacher_prompt_file) as _f:
            _teacher_prompt_override = _f.read().strip()
        rank0_print(f"Teacher prompt override loaded from: {opd_args.teacher_prompt_file}")

    # --- Dataset ---
    dataset = TVGOPDDataset(
        processor=processor,
        data_args=data_args,
        training_mode="grpo",
        teacher_cache_dir=opd_args.teacher_cache_dir,
        teacher_strategy=opd_args.teacher_strategy,
        student_mark_strategy=opd_args.student_mark_strategy,
        ref_anno_path=opd_args.ref_anno_path,
        teacher_prompt_override=_teacher_prompt_override,
    )

    # --- Trainer ---
    rank0_print(
        f"OPD config: kl_weight={opd_args.kl_weight}, "
        f"grpo_weight={opd_args.grpo_weight}, "
        f"kl_temperature={opd_args.kl_temperature}, "
        f"jsd_alpha={opd_args.jsd_alpha}, "
        f"distillation_topk={opd_args.distillation_topk}, "
        f"teacher_strategy={opd_args.teacher_strategy}, "
        f"teacher_quality_gate={opd_args.teacher_quality_gate}"
    )

    trainer = OPDGRPOTrainer(
        model=model,
        processing_class=processor,
        train_dataset=dataset,
        reward_funcs=reward_funcs_list,
        args=training_args,
        kl_weight=opd_args.kl_weight,
        grpo_weight=opd_args.grpo_weight,
        kl_temperature=opd_args.kl_temperature,
        jsd_alpha=opd_args.jsd_alpha,
        distillation_topk=opd_args.distillation_topk,
        teacher_quality_gate=opd_args.teacher_quality_gate,
        kl_weight_end=opd_args.kl_weight_end,
        kl_schedule=opd_args.kl_schedule,
        divergence=opd_args.divergence,
        jsd_clip_max=opd_args.jsd_clip_max,
        exopd_lambda=opd_args.exopd_lambda,
        entropy_gate=opd_args.entropy_gate,
        entropy_tau=opd_args.entropy_tau,
        golden_keep_frac=opd_args.golden_keep_frac,
        two_stage_explore_frac=opd_args.two_stage_explore_frac,
        exopd_ref_path=opd_args.exopd_ref_path,
        frozen_teacher=frozen_teacher,
        teacher_update_strategy=opd_args.teacher_update_strategy,
        ema_decay=opd_args.ema_decay,
        dead_rescue_enabled=opd_args.dead_rescue_enabled,
        dead_rescue_mode=opd_args.dead_rescue_mode,
        dead_reward_std_eps=opd_args.dead_reward_std_eps,
        dead_reward_mean_threshold=opd_args.dead_reward_mean_threshold,
        dead_group_criterion=opd_args.dead_group_criterion,
        teacher_advantage_margin=opd_args.teacher_advantage_margin,
        dead_rescue_multiplier=opd_args.dead_rescue_multiplier,
        negative_teacher_multiplier=opd_args.negative_teacher_multiplier,
        # 新增开关 (默认值 = 原行为)
        teacher_tiou_mode=opd_args.teacher_tiou_mode,
        teacher_gen_max_new_tokens=opd_args.teacher_gen_max_new_tokens,
        teacher_gate_mode=opd_args.teacher_gate_mode,
        dead_rescue_norm=opd_args.dead_rescue_norm,
        jsd_token_focus=opd_args.jsd_token_focus,
        jsd_numeric_weight=opd_args.jsd_numeric_weight,
        # 论文 Eq5/Eq6 组级 controller + teacher 独立 rollout
        adaptive_controller=opd_args.adaptive_controller,
        controller_lambda_no_edge=opd_args.controller_lambda_no_edge,
        controller_lambda_dead=opd_args.controller_lambda_dead,
        controller_skip_zero=opd_args.controller_skip_zero,
        teacher_rollout_mode=opd_args.teacher_rollout_mode,
        teacher_rollout_n=opd_args.teacher_rollout_n,
        teacher_rollout_batch=opd_args.teacher_rollout_batch,
    )

    # --- 早段checkpoint即存即删 (2026-09-15, v2-s225实验): ---
    # 需求: 只保留 global_step >= 75 的存档 (75/100/.../225), 25/50 写完立即删除,
    # 避免每个16G白白占盘。on_save在HF保存流程完成后触发; 仅主进程有文件, 加should_save守卫。
    # 不涉及集合通信 -> 无需barrier; save_total_limit轮换只按目录数, 早段已删故只算75+。
    from transformers import TrainerCallback as _TC
    import shutil as _shutil

    _SAVE_FROM_STEP = 75

    class _DropEarlyCheckpointCallback(_TC):
        def on_save(self, args, state, control, **kw):
            if state.global_step < _SAVE_FROM_STEP and args.should_save:
                ckpt = _os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                if _os.path.isdir(ckpt):
                    _shutil.rmtree(ckpt, ignore_errors=True)
                    rank0_print(f"[DropEarlyCkpt] step {state.global_step} < {_SAVE_FROM_STEP}, 已删除")

    trainer.add_callback(_DropEarlyCheckpointCallback())

    # --- Train ---
    # 设置逐样本诊断日志路径
    import os as _os
    _diag_dir = _os.path.join(training_args.output_dir, "diagnostics")
    _os.makedirs(_diag_dir, exist_ok=True)
    trainer._diag_jsonl_path = _os.path.join(_diag_dir, "train_groups.jsonl")
    rank0_print(f"Diagnostics JSONL: {trainer._diag_jsonl_path}")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
# PATCH: zoom_expand already added on old machine, verify below
