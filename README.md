# DTP / OPSD 训练代码存档

本仓库存放两个最佳 checkpoint 对应的训练代码快照。

## 目录对应关系

| 目录 | 对应 run | 评测 (Charades mIoU) | 说明 |
|---|---|---|---|
| `v3paper/` | `fixed-all-152-v3paper-s225` @ ckpt-100 | **55.93** 🥇 | v3 论文公式版（controller 未生效） |
| `v4real-m005/` | `v4real-m005-20260919-2029` @ ckpt-125 | **55.87** 🥈 | v4：controller 真正生效（forced 组级估计 + margin=-0.05） |

## 方法概要

OPD（On-policy Privileged Distillation）= GRPO + 特权教师蒸馏：

- teacher 看带红框标记目标区间的特权视频（`coarse_box_multi`，zoom_expand=1.5），student 看干净视频
- JSD 蒸馏 + 组级 adaptive controller（Eq.5：teacher 无优势的组 λ=0 不蒸馏，dead group 有优势 λ=2 加倍）
- v4 相对 v3 的修复：controller 的 r̄T 改用 teacher-forced 组级估计（shuffle 前聚合），skip_zero 全卡一致化防 NCCL 死锁

## 关键训练配置（两版一致）

- 基座：`TimeLens-Qwen3-VL-8B-SFT`
- 数据：`train_reviewed_152.jsonl`（152 条，Charades 来源）
- 8×H20，deepspeed zero2_offload，`num_generations=8`，`steps_per_generation=4`，lr=1e-6 constant，`max_steps=225`

## 依赖

- TimeLens 基座代码：https://github.com/JungleGym/TimeLens
- 环境（H20 + python3.12 venv）：torch 2.x / transformers / trl 0.19.1 / deepspeed 0.19.2 / decord / qwen-vl-utils / flash-attn

## 训练入口

```bash
bash scripts/run_opd_tvg_h20_v4_controller.sh   # v4real-m005
# v3paper 见 v3paper/scripts/ 与 README_V3_SNAPSHOT.md
```

各子目录内的 README（`README_V3_SNAPSHOT.md` / `README_V4.md`）有该版的详细实验记录。
