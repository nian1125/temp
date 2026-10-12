#!/usr/bin/env bash
# ============================================================================
# DTP-OPSD 主实验 v3 = 论文对齐版 (paper-faithful)
#
# 目标: 让代码严格等于 DTP_OPSD_ICLR2027.tex 里 Eq(3)-(6) 所描述的方法。
# v1 (run_opd_tvg_h20_fixed_all_152.sh, 论文 55.62) 与论文文本有 5 处实质差异,
# v3 全部对齐。v1/v2 脚本均保留不变。
#
# --------------------------------------------------------------------------
# [D1] 组级 adaptive controller (论文 Eq5)  → --adaptive_controller paper
#   论文:  λ_g = 0 若 r̄_g^T ≤ r̄_g^S ; = 2 若 r̄_g^T > r̄_g^S 且组内全零 ; 否则 1
#          逐【组】判定, 再按 Eq6 的 (1/|B|)Σ_g λ_g ℓ_g 归一。
#   v1  :  teacher_advantage 取整 batch 的 .mean() 得到一个标量去缩放整个 jsd_loss,
#          且该逻辑只在 dead_rescue 分支内生效 → batch 级近似, 非论文口径。
#          更严重的是 v1 里它根本没执行(见 D2)。
#
# [D2] teacher 无优势时 λ=0 (论文) vs ×0.25 (v1)
#   论文明确 "λ=0 disables distillation for that group"。
#   实测 teacher_worse_frac≈0.5 → 论文是把这一半组彻底屏蔽, v1 是留 1/4 权重。
#   注: v1 里这段是死代码(父类不返回 rewards → teacher_advantage 恒 None,
#   日志 teacher_advantage_mean 全程 0.0), 所以 v1 实际连 0.25 都没做到, 是 ×1。
#
# [D3] teacher competence 用独立采样 rollout (论文) → --teacher_rollout_mode sample
#   论文: "independently generated teacher rollouts under the same sampling
#   settings (temperature, G)"。
#   v1: _estimate_teacher_tiou 在 student 采样出的【前缀】上取 teacher-forced argmax。
#   时间数字是逐位生成的, 前缀一旦被 student 锚定, teacher 只能顺着 student 的答案
#   往下补 → 系统性偏向 student, 既高估 top1 一致率(0.98)也测不出 teacher 真实能力。
#
# [D4] dead group 判据 = all-zero (论文) → --dead_group_criterion all_zero
#   论文: r_{g,i} = 0 ∀i。v1: std≈0 且 mean<0.30 → 全组同为 0.25 会被误判成 dead。
#
# [D5] 蓝框保留 verified 边界 (论文) → --blue_zoom_expand 1.0
#   论文: "Distractor windows retain their verified boundaries", 只有红框扩 1.5×。
#   v1: 蓝框也乘了 zoom_expand=1.5 (与红框共用 _cexp), 向两侧溢出后可能盖到红框
#   附近, 使"哪里不是目标"这条干扰信息本身变得不可靠。这是纯 bug。
#
# --------------------------------------------------------------------------
# 另外保留 v2 的两项工程修正 (论文未提, 但 v1 的实现使机制哑火):
#   --batch_per_device 2   : v1 的 loss batch=1 会让任何逐样本加权在归一化后抵消
#   (v3 的 controller 用 .mean() 归一, 不受该 bug 影响, 但 batch>1 仍降低梯度方差)
#
# 故意【不】开启的 v2 选项:
#   --jsd_token_focus : 论文 Eq4 是在全部 valid completion token 上平均。开了会偏离
#                       论文文本。若消融确认有效, 需要往论文里补一句再开。
#
# 用法:
#   nohup bash scripts/run_opd_tvg_h20_fixed_all_152_v3_paper.sh \
#       >> /dockerdata/zhenjunchen/logs/run_fixed_all_152_v3.log 2>&1 &
#
# 成本提示: teacher rollout (G=8 条采样) 每个 generation block 多一次生成,
#   但 controller 让约一半的组 λ=0 → --controller_skip_zero True 会跳过那些切片的
#   teacher forward。两者大致相抵, 预计 step 时间与 v1 同量级(±20%)。
#   若 OOM: 先调 --teacher_rollout_batch 4 (或 2), 再考虑 BATCH_PER_DEVICE=1。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANNO="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_152.jsonl"

# ---- 可调: OOM 时依次尝试 TEACHER_ROLLOUT_BATCH=4 → 2, 再 BATCH_PER_DEVICE=1/GLOBAL_BATCH=8 ----
BATCH_PER_DEVICE=2
GLOBAL_BATCH=32
TEACHER_ROLLOUT_BATCH=0     # 0=一次生成G条; OOM时设 4 或 2
# ------------------------------------------------------------------------------------------------

if pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; then
  echo "[ERROR] 检测到已有训练进程在运行:"
  pgrep -af "train_opd_tvg.py"
  exit 1
fi

echo ">>> 数据预检: ${ANNO}"
python3 << 'PYEOF'
import json, os, sys, collections
anno='/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_152.jsonl'
missing=0; total=0; badspan=0; rows=[]
with open(anno) as f:
    for line in f:
        d=json.loads(line); total+=1; rows.append(d)
        vp=d['video_path']
        if not vp.startswith('/'):
            vp=os.path.join('/dockerdata/zhenjunchen/train_videos', vp)
        if not os.path.exists(vp):
            missing+=1
            if missing<=5: print("  MISS:", vp)
        sp=d.get('span',[])
        if not sp or sp[0][0]>=sp[0][1] or sp[0][1]>d.get('duration',1e9)+0.01:
            badspan+=1
# 蓝框覆盖率: coarse_box_multi 的干扰信息只有同视频有 >1 事件时才存在
c=collections.Counter(d['video_path'] for d in rows)
blue=sum(1 for d in rows if c[d['video_path']]>1)
print(f"预检: {total}条, 视频缺失={missing}, span异常={badspan}")
print(f"蓝框可用样本: {blue}/{total} ({100.0*blue/max(total,1):.1f}%)  "
      f"[用训练集自身当ref; 论文口径应 --ref_anno_path 指全量标注]")
if missing>0 or badspan>0:
    print("FATAL: 数据预检未通过, 中止启动"); sys.exit(1)
print("预检通过 ✅")
PYEOF
if [ $? -ne 0 ]; then exit 1; fi

VENV_PATH="/dockerdata/zhenjunchen/venv"
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# teacher 视频 cache_key 依赖 other_spans 的顺序, 代码已 sorted(); 固定 hashseed 双保险
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
echo "DTP-OPSD 主实验 v3 (论文 Eq3-6 对齐版)"
echo "数据: train_reviewed_152.jsonl (152条, 已预检)"
echo "D1 组级controller: adaptive_controller=paper (λ∈{0,1,2} 逐组)"
echo "D2 teacher无优势: λ=0 (完全不蒸馏, 非×0.25)"
echo "D3 teacher能力  : 独立采样 G=8 条 (非 student前缀上的argmax)"
echo "D4 dead判据     : all_zero (论文 r=0 ∀i)"
echo "D5 蓝框         : blue_zoom_expand=1.0 (保留verified边界, 只扩红框)"
echo "工程            : batch_per_device=${BATCH_PER_DEVICE}"
echo "H20 8卡"
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
    --max_steps 225 \
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
    --teacher_rollout_mode sample \
    --teacher_rollout_n 0 \
    --teacher_rollout_batch ${TEACHER_ROLLOUT_BATCH} \
    --total_tokens 14336 \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2_offload_full.json" \
    --report_to "none" \
    --teacher_prompt_file "${PROJECT_ROOT}/configs/prompt_student_same.txt" \
    --name_suffix "-fixed-all-152-v3paper-s225"

echo "fixed_all+152 v3(论文对齐) 训练完成: $(date)"
