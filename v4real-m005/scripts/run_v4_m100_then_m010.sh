#!/usr/bin/env bash
# ============================================================================
# 串行: [1/2] v4-m100 (margin=-1.0, ≈全程蒸馏)  ->  [2/2] v4-m010 (margin=-0.10)
#
# 动机: 蒸馏充分性与评测效果呈单调正相关 —
#   forced(margin=0,   λ=1占37%) 评测差
#   gen   (margin=0,   λ=1占31%) 评测差
#   m005  (margin=-.05, λ=1占68%) 评测最好 ✅
# 继续向"蒸馏更充分"方向推进, 两个实验补全 margin 曲线:
#
#   [1/2] m100: margin=-1.0 → no_edge = (r̄T <= r̄S - 1.0) 几乎永不成立
#         → λ∈{1,2}, controller 不再过滤 (仅 dead group 仍 λ=2)
#         回答: "过滤 teacher 差组"到底有没有用?
#         若 m100 最好 → 过滤多余, 最终配置就是固定蒸馏;
#         若 m100 退步 → 过滤 teacher 远差组(占18%)有价值, 甜点在 -0.05~-0.10。
#   [2/2] m010: margin=-0.10 → 只过滤 teacher 远差(≤-0.15)的组, 预计 λ=0 降至 ~18%
#
# 机制说明: margin/name_suffix 经环境变量传入 run_opd_tvg_h20_v4_controller.sh
#   (TEACHER_MARGIN / NAME_SUFFIX), 串行过程不需要编辑任何训练脚本。
#
# ★ 全部逻辑包在 main() 里最后一行才调用: bash 先完整解析函数体,
#   运行期编辑本文件不影响执行。即便如此【运行期间请勿编辑相关脚本】。
#
# 用法:
#   nohup bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4/scripts/run_v4_m100_then_m010.sh \
#       >> /mnt/250T_ceph/zhenjunchen/logs/v4_m100_then_m010.log 2>&1 &
# 预期: 每步 ~9h, 共 ~18h。forced 路径不生成, 无 hang 风险。
# ============================================================================
set -uo pipefail

main() {
    local V4=/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4
    local rc1=0 rc2=0

    echo "############################################################"
    echo "# 串行任务启动: $(date)"
    echo "# [1/2] v4-m100  (margin=-1.0, ≈全程蒸馏)"
    echo "# [2/2] v4-m010  (margin=-0.10, 只过滤teacher远差组)"
    echo "############################################################"

    echo ""
    echo "===== [1/2] v4-m100 开始: $(date) ====="
    TEACHER_MARGIN=-1.0 NAME_SUFFIX="-v4real-m100" \
        bash "${V4}/scripts/run_opd_tvg_h20_v4_controller.sh" || rc1=$?
    echo "===== [1/2] v4-m100 结束: $(date) (rc=${rc1}) ====="

    echo ">>> 等待 180s 让 GPU 与 NCCL 资源释放 ..."
    sleep 180
    pkill -9 -f "train_opd_tvg.py" 2>/dev/null || true
    sleep 20

    echo ""
    echo "===== [2/2] v4-m010 开始: $(date) ====="
    TEACHER_MARGIN=-0.10 NAME_SUFFIX="-v4real-m010" \
        bash "${V4}/scripts/run_opd_tvg_h20_v4_controller.sh" || rc2=$?
    echo "===== [2/2] v4-m010 结束: $(date) (rc=${rc2}) ====="

    echo ""
    echo "############################################################"
    echo "# 全部完成: $(date)"
    echo "#   [1/2] v4-m100 rc=${rc1}"
    echo "#   [2/2] v4-m010 rc=${rc2}"
    echo "############################################################"
}

main "$@"
