#!/usr/bin/env bash
# ============================================================================
# DTP-OPSD 主实验 v2 = fixed_all 全配置 + 270 数据, 但把三处失效机制修好
#
# 与 run_opd_tvg_h20_fixed_all_152.sh (v1, 论文 55.62) 的差异【只有下面 4 项】:
#
#   [修2] loss batch 从 1 提到 2 (--batch_per_device 2)
#         + --dead_rescue_norm mean
#         v1 里 dead-rescue 是恒等变换: fixed 模式做 sum(j*w)/sum(w), 而 loss batch=1
#         (global 8 / bpd 1 / 8卡 → grad_accum 1, spg 4 把 4 条切成 4 份 → 每次只有1条),
#         w 被自己的分母除掉。实证: v1 日志里 jsd_total_weighted 与 jsd_total_raw
#         逐段完全相等, 14 步"触发"rescue 但 loss 一点没变。
#         norm=mean 让死组 JSD 真正 ×multiplier; bpd=2 让"组内相对放大"语义成立。
#
#   [修3] --teacher_tiou_mode generate + --teacher_gate_mode on
#         v1 的 teacher 质量门控是死代码: 父类 _generate_and_score_completions
#         不返回 "rewards" 键 → inputs.get("rewards") 恒 None → teacher_advantage
#         永远算不出来(v1 日志 teacher_advantage_mean 全程 0.0)。已在父类补该键。
#         同时 teacher tIoU 从 "student前缀上的 teacher-forced argmax"(被 student
#         锚定, 系统性偏向 student 答案) 换成 teacher 自主贪心解码, 每 rollout block
#         只算一次并随 shuffle 对齐。
#
#   [修4] --jsd_token_focus numeric_soft --jsd_numeric_weight 4.0
#         completion 平均 18 token, 其中约一半是 "The event happens in ... seconds."
#         固定模板, 师生必然一致(v1 top1_agree=0.98, JSD raw≈0.0017)。
#         把散度权重压到真正携带答案的数值 token 上。
#
# 其余超参与 v1 完全一致(coarse_box_multi 红蓝框 / zoom 1.5 / drifting /
# dead rescue fixed / 师生同 prompt / kl 1.0 / gw 1.0 / grpo / spg 4 / lr 1e-6)。
#
# 用法:
#   nohup bash scripts/run_opd_tvg_h20_fixed_all_152_v2.sh \
#       >> /dockerdata/zhenjunchen/logs/run_fixed_all_270_v2.log 2>&1 &
#
# 显存回退方案: 若 bpd=2 OOM, 把 BATCH_PER_DEVICE 改回 1 并把 GLOBAL_BATCH 改回 8。
#   dead_rescue_norm=mean 在 batch=1 时依然有效(等价于死组步单独提高 kl_weight),
#   [修3][修4] 也不受 batch 影响 → 三处修正仍全部生效, 只是 rescue 少了组内相对性。
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANNO="/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_balanced_270_human_charades.jsonl"

# ---- 可调: OOM 时改这两行 (BATCH_PER_DEVICE=1 / GLOBAL_BATCH=8) ----
BATCH_PER_DEVICE=2
GLOBAL_BATCH=32
# --------------------------------------------------------------------

if pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; then
  echo "[ERROR] 检测到已有训练进程在运行:"
  pgrep -af "train_opd_tvg.py"
  exit 1
fi

# ============================================================
# 数据预检 (防静默丢数据): 模拟dataset路径解析, 有缺失则中止
# ============================================================
echo ">>> 数据预检: ${ANNO}"
python3 << 'PYEOF'
import json, os, sys
anno='/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/output/2d_sets/train_reviewed_balanced_270_human_charades.jsonl'
missing=0; total=0; badspan=0
with open(anno) as f:
    for line in f:
        d=json.loads(line); total+=1
        vp=d['video_path']
        if not vp.startswith('/'):
            vp=os.path.join('/dockerdata/zhenjunchen/train_videos', vp)
        if not os.path.exists(vp):
            missing+=1
            if missing<=5: print("  MISS:", vp)
        sp=d.get('span',[])
        if not sp or sp[0][0]>=sp[0][1] or sp[0][1]>d.get('duration',1e9)+0.01:
            badspan+=1
print(f"预检: {total}条, 视频缺失={missing}, span异常={badspan}")
if missing>0 or badspan>0:
    print("FATAL: 数据预检未通过, 中止启动")
    sys.exit(1)
print("预检通过 ✅")
PYEOF
if [ $? -ne 0 ]; then exit 1; fi

VENV_PATH="/dockerdata/zhenjunchen/venv"
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
echo "DTP-OPSD 主实验 v2 (三处失效机制修复版)"
echo "数据: train_reviewed_balanced_270_human_charades.jsonl (270条, 已预检)"
echo "配置: coarse_box_multi红蓝框 + drifting + fixed rescue + 师生同prompt"
echo "修2: batch_per_device=${BATCH_PER_DEVICE} (loss batch>1) + dead_rescue_norm=mean"
echo "修3: teacher_tiou_mode=generate + teacher_gate_mode=on"
echo "修4: jsd_token_focus=numeric_soft (w=4.0)"
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
    --teacher_update_strategy drifting \
    --dead_rescue_enabled True \
    --dead_rescue_mode fixed \
    --dead_reward_std_eps 1e-6 \
    --dead_reward_mean_threshold 0.30 \
    --teacher_advantage_margin 0.05 \
    --dead_rescue_multiplier 2.0 \
    --negative_teacher_multiplier 0.25 \
    --dead_rescue_norm mean \
    --teacher_tiou_mode generate \
    --teacher_gen_max_new_tokens 48 \
    --teacher_gate_mode on \
    --jsd_token_focus numeric_soft \
    --jsd_numeric_weight 4.0 \
    --total_tokens 14336 \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2_offload_full.json" \
    --report_to "none" \
    --teacher_prompt_file "/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/configs/prompt_student_same.txt" \
    --name_suffix "-fixed-all-270-v2-s225"

echo "fixed_all+152 v2 训练完成: $(date)"
