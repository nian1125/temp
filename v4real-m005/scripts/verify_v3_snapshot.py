#!/usr/bin/env python3
"""v3 快照隔离性与完整性校验 (纯静态检查, 不需要 GPU/torch)。

用法:
    python3 /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v3/scripts/verify_v3_snapshot.py

校验三件事:
  1. 隔离: 快照的 PROJECT_ROOT 指向自己, 不会串到主线代码
  2. 冻结: v4 的新增能力在快照里【确实不存在】(否则就不是 v3 了)
  3. 完整: v3 该有的能力 (controller/blue_zoom/all_zero) 都在
"""
import os
import sys

V3 = "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v3"
MAIN = "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ"
# 开发侧路径回退 (在 /apdcephfs 上直接跑时)
if not os.path.isdir(V3):
    V3 = "/apdcephfs/cq_250t_ceph/zhenjunchen/code/Temporal-ZwZ-v3"
    MAIN = "/apdcephfs/cq_250t_ceph/zhenjunchen/code/Temporal-ZwZ"

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
res = []


def chk(name, cond, detail=""):
    res.append(bool(cond))
    print(f"  [{OK if cond else BAD}] {name}" + (f"  {detail}" if detail else ""))


def rd(rel, root=None):
    p = os.path.join(root or V3, rel)
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8") as f:
        return f.read()


print("=" * 70)
print(f"v3 快照校验: {V3}")
print("=" * 70)

print("\n[1] 隔离性")
sh = rd("scripts/run_opd_tvg.sh")
chk("run_opd_tvg.sh 的 PROJECT_ROOT 指向 v3 快照",
    'PROJECT_ROOT="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v3"' in sh)
chk("不残留指向主线的 PROJECT_ROOT",
    'PROJECT_ROOT="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ"' not in sh)
v3sh = rd("scripts/run_opd_tvg_h20_fixed_all_152_v3_paper.sh")
chk("v3 启动脚本用 $(dirname $SCRIPT_DIR) 自动推导 root",
    'PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"' in v3sh)
chk("teacher_prompt_file 指向快照自己的 configs",
    '--teacher_prompt_file "${PROJECT_ROOT}/configs/' in v3sh)
chk("configs/prompt_student_same.txt 已随快照复制",
    os.path.exists(os.path.join(V3, "configs/prompt_student_same.txt")))
for pkg in ("zwz/__init__.py", "zwz/trainer/__init__.py",
            "zwz/data/__init__.py", "zwz/train/__init__.py"):
    chk(f"包结构完整: {pkg}", os.path.exists(os.path.join(V3, pkg)))

print("\n[2] 冻结性 (v4 的新增能力必须【不】存在)")
tr = rd("zwz/trainer/opd_grpo_trainer.py")
ds = rd("zwz/data/tvg_opd_dataset.py")
tt = rd("zwz/train/train_opd_tvg.py")
chk("无 _repeat_visual_kwargs (v4 的 rollout 崩溃修复)",
    "_repeat_visual_kwargs" not in tr)
chk("teacher rollout 仍用 num_return_sequences (v3 的 bug 原样保留)",
    "num_return_sequences=nb" in tr)
chk("无 generation_config=gcfg 复用路径", "generation_config=gcfg" not in tr)
chk("无 _teacher_gen_shape_logged", "_teacher_gen_shape_logged" not in tr)
chk("dataset 无 zoom_jitter", "zoom_jitter" not in ds)
chk("dataset 未 import random", "\nimport random" not in ds)
chk("train_opd_tvg 无 zoom_jitter", "zoom_jitter" not in tt)
chk("run_opd_tvg.sh 无 --zoom_jitter", "--zoom_jitter" not in sh)
chk("run_opd_tvg.sh 无 --max_grad_norm", "--max_grad_norm" not in sh)

print("\n[3] 完整性 (v3 该有的能力)")
chk("组级 controller (adaptive_controller)", "adaptive_controller" in tr)
chk("controller 用 Eq6 的 .mean() 归一",
    "jsd_loss = (per_sample_jsd * lam_s).mean()" in tr)
chk("teacher 独立 rollout + 跨进程归并", "gather_object" in tr)
chk("λ 全0 跳过 teacher forward", "_skip_by_ctrl" in tr)
chk("dead_group_criterion=all_zero 支持", "all_zero" in tr)
chk("dead_rescue_norm 开关", "dead_rescue_norm" in tr)
chk("蓝框独立扩张 blue_zoom_expand", "_bexp" in ds)
chk("窗口 clip 到 [0,duration]", "_gt_lo_c, _gt_hi_c = _clip(" in ds)
chk("other_spans 稳定排序", "_other_spans = sorted(_other_spans)" in ds)

print("\n[4] 守卫 (不属于 v3 的脚本已 stub 化)")
for f in ("scripts/run_opd_tvg_h20_fixed_all_152_v4.sh",
          "scripts/probe_teacher_channel.py"):
    c = rd(f)
    chk(f"{os.path.basename(f)} 已 stub", "不属于 v3 快照" in c)

print("\n[5] 主线未被破坏 (v4 能力仍在主线)")
if os.path.isdir(MAIN):
    mtr = rd("zwz/trainer/opd_grpo_trainer.py", MAIN)
    mds = rd("zwz/data/tvg_opd_dataset.py", MAIN)
    chk("主线有 _repeat_visual_kwargs", "_repeat_visual_kwargs" in mtr)
    chk("主线有 zoom_jitter", "zoom_jitter" in mds)
    chk("主线已去掉 num_return_sequences=nb", "num_return_sequences=nb" not in mtr)
else:
    print("  (跳过: 主线目录不可见)")

n, tot = sum(res), len(res)
print("\n" + "=" * 70)
print(f"结果: {n}/{tot} 通过" + ("" if n == tot else "  ⚠️ 有失败项, 快照不可信!"))
print("=" * 70)
sys.exit(0 if n == tot else 1)
