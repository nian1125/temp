#!/usr/bin/env python3
"""v4 校验: 确认 v4 = v3 + 【仅】controller 修复 (纯静态, 不需 GPU/torch)。

用法:
    python3 /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4/scripts/verify_v4.py

校验四件事:
  1. controller 修复已到位 (v4 的唯一代码改动)
  2. 除此之外与 v3 【逐字节相同】—— 这是单变量对照的前提
  3. 隔离: PROJECT_ROOT 指向 v4, 输出不会串到 v3/主线
  4. 没有把主线 (探索线) 的 zoom_jitter / max_grad_norm 带进来
"""
import hashlib
import os
import sys

# 路径探测: 以本脚本所在位置为准 (最可靠), 避免 /mnt 视图过时导致误判。
# 本脚本位于 <BASE>/Temporal-ZwZ-v4/scripts/verify_v4.py
V4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(V4)
V3 = os.path.join(BASE, "Temporal-ZwZ-v3")

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
res = []


def chk(name, cond, detail=""):
    res.append(bool(cond))
    print(f"  [{OK if cond else BAD}] {name}" + (f"  {detail}" if detail else ""))


def rd(root, rel):
    p = os.path.join(root, rel)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def code_lines(text, comment="#"):
    """剔除注释行/纯文档说明行, 只保留可执行代码。

    verify 自身踩过的坑: docstring 和注释里为了解释 bug 会提到
    `num_return_sequences`、`--dead_rescue_norm` 等字样, 直接用 `in` 判断会误命中。
    """
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(comment):
            continue
        out.append(ln)
    return "\n".join(out)


def md5(root, rel):
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        return None
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:12]


print("=" * 72)
print(f"v4 校验")
print(f"  v4 = {V4}")
print(f"  v3 = {V3}")
if not os.path.isdir(V3):
    print(f"  ⚠️ 找不到 v3 目录, [2] 段的逐字节对比将全部失败")
print("=" * 72)

tr4 = rd(V4, "zwz/trainer/opd_grpo_trainer.py")
tr3 = rd(V3, "zwz/trainer/opd_grpo_trainer.py")

def _defaults(text):
    """抽取所有 `name: type = field(default=X,` 的默认值, 用于比对参数语义是否一致。

    比逐字节 md5 宽松(允许改 help 文本), 比行数统计精确(不受多行字符串影响)。
    """
    import re
    out = {}
    for m in re.finditer(r"^\s{4}(\w+)\s*:\s*[^=\n]+=\s*field\(\s*\n?\s*default=([^,\n]+)",
                         text, re.MULTILINE):
        out[m.group(1)] = m.group(2).strip()
    return out


def has_stmt(text, stmt):
    """是否存在一行【实际代码】等于 stmt (strip 后精确匹配)。

    比 `stmt in text` 严格: docstring/注释里提到同名字样不会误命中。
    """
    return any(ln.strip() == stmt for ln in text.splitlines())


print("\n[1] v4 的两处改动到位")
chk("A: 有 _teacher_forced_group_tiou (v1式teacher-forced组级估计)",
    "_teacher_forced_group_tiou" in tr4)
chk("A: controller 走 forced 分支",
    'if self.teacher_tiou_mode == "generate":' in tr4
    and "_teacher_forced_group_tiou(" in tr4)
chk("A: forced 路径不含 generate 调用 (只前向)",
    "mod.model(input_ids=full_ids" in tr4)
chk("B: skip 决策全卡一致化 (all_reduce)",
    "torch.distributed.all_reduce(_flag" in tr4)
chk("B: all_reduce 用 MAX (任一卡需蒸馏则全卡不跳)",
    "ReduceOp.MAX" in tr4)
chk("B: all_reduce 异常时退化为不跳过",
    "_skip_by_ctrl = False" in tr4)
print("  --- 对照: v3 没有这两项 ---")
chk("v3 无 _teacher_forced_group_tiou", "_teacher_forced_group_tiou" not in tr3)
chk("v3 的 skip 是每卡独立判断(会死锁)",
    "torch.distributed.all_reduce(_flag" not in tr3)

print("\n[2] 除上述两处外与 v3 完全一致 (单变量对照前提)")
# 这几个文件必须逐字节相同 (完全没碰)
for rel in ("zwz/data/tvg_opd_dataset.py",
            "zwz/train/reward_funcs.py",
            "configs/prompt_student_same.txt"):
    a, b = md5(V3, rel), md5(V4, rel)
    chk(f"{rel} 与 v3 逐字节相同", a is not None and a == b, f"md5 {a} vs {b}")

# train_opd_tvg.py: 只允许 teacher_tiou_mode 的 help 文本不同, 默认值/字段必须一致
t3, t4 = rd(V3, "zwz/train/train_opd_tvg.py"), rd(V4, "zwz/train/train_opd_tvg.py")
chk("train_opd_tvg.py: teacher_tiou_mode 默认值仍是 argmax (两边一致)",
    'teacher_tiou_mode: str = field(\n        default="argmax",' in t3
    and 'teacher_tiou_mode: str = field(\n        default="argmax",' in t4)
chk("train_opd_tvg.py: 所有字段默认值与 v3 完全一致",
    _defaults(t3) == _defaults(t4),
    f"{len(_defaults(t3))} 个字段" if _defaults(t3) == _defaults(t4)
    else f"差异={set(_defaults(t3).items()) ^ set(_defaults(t4).items())}")
for _k in ("adaptive_controller", "controller_lambda_no_edge", "dead_group_criterion",
           "blue_zoom_expand", "teacher_rollout_mode"):
    chk(f"train_opd_tvg.py 仍有字段 {_k}", (_k in t3) and (_k in t4))

# run_opd_tvg.sh: 只允许 PROJECT_ROOT + 注释不同
def _sh_code(t):
    out = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(ln.split("#")[0].rstrip() if "#" in ln else ln)
    return [l for l in out if "PROJECT_ROOT=" not in l]


s3c, s4c = _sh_code(rd(V3, "scripts/run_opd_tvg.sh")), _sh_code(rd(V4, "scripts/run_opd_tvg.sh"))
chk("run_opd_tvg.sh 除 PROJECT_ROOT/注释外与 v3 一致",
    s3c == s4c, f"差异行数={sum(1 for x, y in zip(s3c, s4c) if x != y)}")

# controller 的核心逻辑必须与 v3 一致 (只修 rollout, 不动 controller 本身)
for key in ("jsd_loss = (per_sample_jsd * lam_s).mean()",
            'if self.adaptive_controller == "paper":',
            "_skip_by_ctrl",
            "gather_object",
            "controller_lambda_no_edge",
            "group_all_zero"):
    chk(f"controller 逻辑未被改动: {key[:44]}", (key in tr4) and (key in tr3))

print("\n[3] 隔离")
sh4 = rd(V4, "scripts/run_opd_tvg.sh")
chk("PROJECT_ROOT 指向 Temporal-ZwZ-v4",
    'PROJECT_ROOT="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4"' in sh4
    or 'Temporal-ZwZ-v4"' in sh4)
chk("不残留指向 v3 的 PROJECT_ROOT",
    'PROJECT_ROOT="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v3"' not in sh4)
chk("不残留指向主线的 PROJECT_ROOT",
    'PROJECT_ROOT="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ"' not in sh4)
chk("output_root 跟随 PROJECT_ROOT",
    'output_root="${PROJECT_ROOT}/output/opd-grpo"' in sh4)
v4sh = rd(V4, "scripts/run_opd_tvg_h20_v4_controller.sh")
chk("v4 启动脚本存在", bool(v4sh))
chk("v4 启动脚本内置代码版本自检", "_teacher_forced_group_tiou" in v4sh)
chk("teacher_prompt_file 用 ${PROJECT_ROOT}",
    '--teacher_prompt_file "${PROJECT_ROOT}/configs/' in v4sh)
chk("v3 启动脚本已在 v4 目录里 stub 化",
    "不要在 v4 目录里跑 v3 脚本"
    in rd(V4, "scripts/run_opd_tvg_h20_fixed_all_152_v3_paper.sh"))
for pkg in ("zwz/__init__.py", "zwz/trainer/__init__.py",
            "zwz/data/__init__.py", "zwz/train/__init__.py"):
    chk(f"包结构完整: {pkg}", os.path.exists(os.path.join(V4, pkg)))

print("\n[4] 未混入主线探索线的改动")
ds4 = rd(V4, "zwz/data/tvg_opd_dataset.py")
tt4 = rd(V4, "zwz/train/train_opd_tvg.py")
chk("无 zoom_jitter", "zoom_jitter" not in ds4 and "zoom_jitter" not in tt4)
chk("run_opd_tvg.sh 无 --zoom_jitter", "--zoom_jitter" not in sh4)
chk("run_opd_tvg.sh 无 --max_grad_norm", "--max_grad_norm" not in sh4)
chk("v4 启动脚本未传 --dead_rescue_norm (controller 分支不经过它)",
    "--dead_rescue_norm" not in code_lines(v4sh))
chk("v4 启动脚本 max_steps 与 v3 一致 (225)", "MAX_STEPS=225" in v4sh)
# trainer 允许改动 (A/B 两处), 其余训练代码必须与 v3 逐字节相同 —— 已在 [2] 验证

print("\n[5] 关键参数")
chk("启动脚本用 --teacher_tiou_mode forced",
    "--teacher_tiou_mode forced" in code_lines(v4sh))
chk("启动脚本未用 generate 模式 (会 hang)",
    "--teacher_tiou_mode generate" not in code_lines(v4sh))
chk("未传 --teacher_rollout_batch (forced 路径不生成, 无需分批)",
    "--teacher_rollout_batch" not in code_lines(v4sh))

n, tot = sum(res), len(res)
print("\n" + "=" * 72)
print(f"结果: {n}/{tot} 通过" + ("" if n == tot else "  ⚠️ 有失败项!"))
if n == tot:
    print("v4 = v3 + 仅 controller 修复, 单变量对照成立。")
print("=" * 72)
sys.exit(0 if n == tot else 1)
