#!/usr/bin/env python3
"""从旧实验启动脚本中提取完整配置，生成 config manifest。

用法:
  python zwz/train/extract_old_config.py
"""
import json
import os
import sys

# 从 exp2/exp3 启动脚本中提取的参数（精确复制自 shell 脚本）
# 来源: scripts/run_opd_tvg_h20_exp2.sh 和 run_opd_tvg_h20_exp3.sh
# 这两个脚本只通过 CLI 覆盖了以下参数，其余全部是 run_opd_tvg.sh 的默认值

def extract_configs():
    """提取两个旧实验的完整配置。"""

    # 公共基础参数（从 run_opd_tvg.sh 的默认值和 deepspeed 启动命令中提取）
    common = {
        # 模型
        "model_name_or_path": "/dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT",
        "model_id": "qwen3-vl-8b-opd",

        # 数据
        "anno_path": "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_839_local.jsonl",
        "video_dir": "/dockerdata/zhenjunchen/train_videos",
        "dataset_name": "filtered_hybrid",

        # OPD-Teacher
        "teacher_strategy": "coarse_box",
        "teacher_cache_dir": "/dockerdata/zhenjunchen/teacher_cache",
        "teacher_quality_gate": False,
        "freeze_teacher": False,
        "exopd_lambda": 1.25,
        "exopd_ref_path": None,
        "student_mark_strategy": None,
        "processor_path": None,

        # OPD-损失
        "kl_weight": 1.0,
        "kl_schedule": "constant",
        "kl_weight_end": None,
        "kl_temperature": 1.0,
        "jsd_alpha": 0.5,
        "jsd_clip_max": 0.0,
        "distillation_topk": 100,
        "divergence": "jsd",
        "entropy_gate": "off",
        "entropy_tau": 0.8,
        "golden_keep_frac": 1.0,
        "two_stage_explore_frac": 0.0,
        "grpo_weight": 1.0,
        "loss_type": "grpo",
        "beta": 0.0,
        "scale_rewards": False,
        "reward_funcs": "tiou",

        # GRPO
        "num_generations": 8,
        "steps_per_generation": 8,
        "temperature": 1.0,
        "max_completion_length": 512,

        # 训练超参
        "learning_rate": 1e-6,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "max_steps": 350,
        "save_steps": 25,
        "logging_steps": 1,
        "save_total_limit": 20,
        "seed": 42,
        "epochs": 1,

        # 模型冻结
        "freeze_vision_tower": True,
        "freeze_llm": False,
        "freeze_merger": False,

        # 精度/并行
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "gradient_checkpointing": True,
        "disable_flash_attn2": False,
        "num_devices": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "dataloader_num_workers": 4,
        "ddp_timeout": 7200,

        # 视频处理
        "min_tokens": 64,
        "total_tokens": 14336,
        "fps": 2,
        "fps_max_frames": 448,
        "min_video_len": 5,
        "max_video_len": 500,

        # DeepSpeed
        "deepspeed": "scripts/zero2_offload_full.json",

        # 报告
        "report_to": "none",
        "use_liger": False,
        "use_liger_loss": False,
        "log_completions": True,
        "save_only_model": True,
        "remove_unused_columns": False,

        # Prompt（从代码中确认）
        "student_prompt": "clean video (no marks)",
        "teacher_prompt": "coarse_box red border (no text prefix)",
    }

    # EXP2 (1.25) 特有
    old_config_125 = dict(common)
    old_config_125["zoom_expand"] = 1.25
    old_config_125["run_name"] = "opd-coarse_box-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260722-2113"

    # EXP3 (1.75) 特有
    old_config_175 = dict(common)
    old_config_175["zoom_expand"] = 1.75
    old_config_175["run_name"] = "opd-coarse_box-kl1.0-gw1.0-grpo-spg8-lr1e-6-20260723-0102"

    return old_config_125, old_config_175


def check_consistency(cfg1, cfg2, ignore_fields=None):
    """检查两个配置的一致性，忽略指定字段。"""
    if ignore_fields is None:
        ignore_fields = {"zoom_expand", "run_name", "output_dir", "experiment_name",
                         "checkpoint_path", "timestamp", "log_dir"}

    matched, different, missing = [], [], []
    all_keys = set(cfg1.keys()) | set(cfg2.keys())

    for key in sorted(all_keys):
        if key in ignore_fields:
            continue
        if key not in cfg1 or key not in cfg2:
            missing.append(key)
        elif cfg1[key] != cfg2[key]:
            different.append((key, cfg1[key], cfg2[key]))
        else:
            matched.append(key)

    return matched, different, missing


def main():
    old_125, old_175 = extract_configs()

    print("=" * 60)
    print("[OLD CONFIG CONSISTENCY CHECK]")
    print("=" * 60)

    matched, different, missing = check_consistency(old_125, old_175)

    print(f"matched_fields = {len(matched)}")
    print(f"different_fields = {len(different)}")
    print(f"missing_fields = {len(missing)}")

    if different:
        print("\n⚠️ 不一致字段:")
        for key, v1, v2 in different:
            print(f"  {key}: 1.25={v1} vs 1.75={v2}")
    else:
        print("✅ 所有非预期字段一致")

    if missing:
        print(f"\n⚠️ 缺失字段: {missing}")

    # 关键参数确认
    print("\n关键优化参数确认:")
    for key in ["learning_rate", "lr_scheduler_type", "warmup_steps",
                "kl_weight", "grpo_weight", "exopd_lambda",
                "freeze_teacher", "num_generations", "steps_per_generation",
                "seed", "max_steps", "save_steps"]:
        print(f"  {key} = {old_125[key]}")

    # 输出 common config
    common = dict(old_125)
    del common["zoom_expand"]
    del common["run_name"]
    print(f"\n[COMMON CONFIG] 共 {len(common)} 个字段")

    # 写入 manifest
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "output", "config_manifests"
    )
    os.makedirs(output_dir, exist_ok=True)

    for name, cfg in [("old_config_125", old_125), ("old_config_175", old_175)]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"写入: {path}")

    common_path = os.path.join(output_dir, "common_old_config.json")
    with open(common_path, "w") as f:
        json.dump(common, f, indent=2, ensure_ascii=False)
    print(f"写入: {common_path}")


if __name__ == "__main__":
    main()
