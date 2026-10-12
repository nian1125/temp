#!/usr/bin/env bash
# ============================================================================
# v4 (真) = v3 + controller 真正生效
#
# 相对 v3 只改两件事, 其余【逐字相同】(225步/weighted/无jitter/默认grad_clip):
#
#   [A] controller 的 r̄^T 改用 v1 的估计方式 —— teacher-forced argmax
#       --teacher_tiou_mode generate
#       teacher 看特权视频, 在【student 已采样的那条 completion】上做一次 no_grad
#       前向, 逐位置取 argmax 解码成时间区间, 与 GT 算 tIoU, 再按组聚合成 r̄_g^T。
#       与 v1 的 _estimate_teacher_tiou 语义完全一致, 只是提前到 shuffle 之前算
#       (shuffle/split 之后凑不出整组, controller 需要组级统计)。
#       ★ 不生成 → 没有 KV cache 增长、没有采样随机性、各卡耗时对称 → 不会 hang。
#
#   [B] 修掉 controller_skip_zero 的集合通信死锁 (代码侧, 见下)
#
# ---------------------------------------------------------------------------
# 【为什么放弃 teacher 独立 rollout】—— 两次实测都失败
#
#   v3 (run_v3_s225.log):
#     generate(num_return_sequences=nb) 让 HF 自己扩 batch, 但它只 expand
#     dim0==batch_size 的 kwargs → video_grid_thw(1,3) 扩了、
#     pixel_values_videos(N,D) 没扩 → split_with_sizes 崩。
#     8 卡静默回退 → [CTRL] 只打印 1 次警告 → λ≡1 → controller 全程没生效。
#
#   v4-rollout (run_v4_controller.log):
#     手工复制 batch 行修好了形状, 日志证明 rollout 真的跑起来了:
#         [TGEN] input_ids=(1, 13667) visual={'pixel_values_videos': (51264,1536),
#                'video_grid_thw': (1,3)} n_roll=8 do_sample=True
#         [CTRL] step=0 r̄T=0.1585 r̄S=0.2753 λ=0:8/8 λ̄=0.000
#     但随后 hang 2 小时被 watchdog 杀:
#         WorkNCCL(SeqNum=19, OpType=_ALLGATHER_BASE, NumelIn=1, NumelOut=8)
#         ran for 7200006 milliseconds
#     两个原因叠加:
#       (1) 8 条 × 13667 token 的 prefill + 逐 token 采样, 每个 block 一次, 太慢;
#       (2) λ=0:8/8 触发 skip_zero, 而被跳过的 JSD 分支里【含 accelerator.gather】,
#           skip 又是每卡独立判断的 → 有的卡跳、有的卡不跳 → 集合通信永久失配。
#           NumelIn=1/NumelOut=8 正是 gather 一个标量的特征。
#
#   → [B] 的修法: skip 决策先做一次 all_reduce(MAX) 全卡一致化, 只有【所有卡】
#     都 λ=0 才整体跳过。all_reduce 本身无条件执行, 不引入新失配。
#     异常时退化为不跳过(宁可多算, 不可死锁)。
#
# ---------------------------------------------------------------------------
# 【诚实记录一个已知偏差】(建议写进论文 limitation)
#   teacher-forced argmax 会被 student 的前缀锚定: 时间数字是逐位生成的, 前缀一旦
#   确定, teacher 只能顺着往下补 → r̄^T 系统性偏向 r̄^S, controller 更容易判
#   "teacher 无优势"(λ=0)。这是 v1 就有的性质, 不是本版新引入的。
#   作为参照: v4-rollout 用【独立采样】测得 r̄T=0.1585 vs r̄S=0.2753 (teacher 明显更差),
#   而 v3 用 argmax 前缀测得 teacher_tiou≈0.49~0.63 vs student≈0.50~0.64 (几乎持平)。
#   两者差距正是"锚定效应"的量级, 说明 forced 估计偏乐观。
#
# ---------------------------------------------------------------------------
# 用法:
#   nohup bash scripts/run_opd_tvg_h20_v4_controller.sh \
#       >> /dockerdata/zhenjunchen/logs/run_v4gen.log 2>&1 &
#
# 开销预期: 相对 v3 每个 generation block 多一次 teacher-forced 前向(本卡 8 条,
#   no_grad)。v1 的 teacher forward 开销约 +12% (37.9s vs 33.7s/步), 本版新增同量级,
#   预计 ~42s/步, 远低于 v3-rollout 的 132.5s/步。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANNO="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_152.jsonl"

# ---- 与 v3 完全一致 ----
BATCH_PER_DEVICE=2
GLOBAL_BATCH=32
MAX_STEPS=225

if pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; then
  echo "[ERROR] 检测到已有训练进程在运行:"
  pgrep -af "train_opd_tvg.py"
  exit 1
fi

# ============================================================
# 启动前自检: 确认代码含 forced 路径 + skip 死锁修复
# ============================================================
echo ">>> 代码版本自检"
python3 - "$PROJECT_ROOT" << 'PYEOF'
import os, sys
root = sys.argv[1]
src = open(os.path.join(root, "zwz/trainer/opd_grpo_trainer.py"), encoding="utf-8").read()
checks = [
    ("v1式 teacher-forced 组级估计", "_teacher_forced_group_tiou" in src),
    ("skip 决策全卡一致化(防NCCL死锁)", "all_reduce(_flag" in src or "_flag, op=torch.distributed.ReduceOp.MAX" in src),
    ("controller 组级 Eq5",            'if self.adaptive_controller == "paper":' in src),
    ("Eq6 归一化 .mean()",             "jsd_loss = (per_sample_jsd * lam_s).mean()" in src),
]
print(f"  代码目录: {root}")
bad = False
for name, ok in checks:
    print(f"  [{'OK ' if ok else 'BAD'}] {name}")
    bad |= (not ok)
if bad:
    print("FATAL: 代码不完整, 检查 PROJECT_ROOT 是否指向 Temporal-ZwZ-v4")
    sys.exit(1)
print("  ✅ 代码校验通过")
PYEOF
if [ $? -ne 0 ]; then exit 1; fi

echo ">>> 数据预检: ${ANNO}"
python3 - "$ANNO" << 'PYEOF'
import json, os, sys, collections
anno = sys.argv[1]
rows=[]; missing=badspan=0
with open(anno) as f:
    for line in f:
        d=json.loads(line); rows.append(d)
        vp=d['video_path']
        if not vp.startswith('/'):
            vp=os.path.join('/dockerdata/zhenjunchen/train_videos', vp)
        if not os.path.exists(vp):
            missing+=1
            if missing<=5: print("  MISS:", vp)
        sp=d.get('span',[])
        if not sp or sp[0][0]>=sp[0][1] or sp[0][1]>d.get('duration',1e9)+0.01:
            badspan+=1
total=len(rows)
c=collections.Counter(d['video_path'] for d in rows)
blue=sum(1 for d in rows if c[d['video_path']]>1)
print(f"预检: {total}条, 视频缺失={missing}, span异常={badspan}")
print(f"蓝框可用: {blue}/{total} ({100.0*blue/max(total,1):.1f}%)")
if missing>0 or badspan>0:
    print("FATAL: 数据预检未通过, 中止启动"); sys.exit(1)
print("预检通过 ✅")
PYEOF
if [ $? -ne 0 ]; then exit 1; fi

VENV_PATH="/dockerdata/zhenjunchen/venv"
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

# NCCL / H20 + RoCE bond
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=INFO

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=========================================="
echo "v4 (真) = v3 + controller 真正生效"
echo "代码: ${PROJECT_ROOT}"
echo "B: teacher_tiou_mode=generate (teacher独立采样, v4修复版)"
echo "B: skip_zero 全卡一致化 (修 v4-rollout 的 NCCL 死锁)"
echo "其余与 v3 逐字相同: max_steps=${MAX_STEPS}, weighted, 无jitter"
echo "=========================================="

bash "${SCRIPT_DIR}/run_opd_tvg.sh" \
    --model_path "/dockerdata/zhenjunchen/models/TimeLens-Qwen3-VL-8B-SFT" \
    --anno_path "${ANNO}" \
    --video_dir "/dockerdata/zhenjunchen/train_videos" \
    --num_devices 8 \
    --kl_weight 1.0 \
    --grpo_weight 1.0 \
    --global_batch_size ${GLOBAL_BATCH} \
    --batch_per_device ${BATCH_PER_DEVICE} \
    --steps_per_generation 4 \
    --loss_type grpo \
    --num_generations 8 \
    --max_steps ${MAX_STEPS} \
    --save_steps 25 \
    --teacher_strategy coarse_box_multi \
    --zoom_expand 1.5 \
    --blue_zoom_expand 1.0 \
    --teacher_update_strategy drifting \
    --dead_rescue_enabled True \
    --dead_rescue_mode fixed \
    --dead_group_criterion all_zero \
    --dead_reward_std_eps 1e-6 \
    --teacher_advantage_margin 0.0 \
    --adaptive_controller paper \
    --controller_lambda_no_edge 0.0 \
    --controller_lambda_dead 2.0 \
    --controller_skip_zero True \
    --teacher_tiou_mode generate \
    --total_tokens 14336 \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2_offload_full.json" \
    --report_to "none" \
    --teacher_prompt_file "${PROJECT_ROOT}/configs/prompt_student_same.txt" \
    --name_suffix "-v4real-gen"

echo "v4(真) 训练完成: $(date)"
