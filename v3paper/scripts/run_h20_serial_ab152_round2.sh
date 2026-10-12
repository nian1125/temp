#!/usr/bin/env bash
# ============================================================
# ab152 第二轮消融串行队列: 补齐5个缺失run
#   1. rkl          (散度消融, 脚本现成, 上次step32被OOM杀)
#   2. jsd_only     (GRPO/JSD消融: 仅蒸馏)
#   3. ema          (教师更新消融: EMA-0.995)
#   4. red_only     (特权信息消融: 仅红框)
#   5. random_blue  (特权信息消融: 红+随机蓝)
# 单个失败不阻塞后续; 每个子脚本自带进程守卫+数据预检
# 用法: nohup bash scripts/run_h20_serial_ab152_round2.sh >> /dockerdata/zhenjunchen/logs/serial_ab152_round2.log 2>&1 &
# ============================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUNS=(
    "run_opd_tvg_h20_ab152_rkl.sh"
    "run_opd_tvg_h20_ab152_jsd_only.sh"
    "run_opd_tvg_h20_ab152_ema.sh"
    "run_opd_tvg_h20_ab152_red_only.sh"
    "run_opd_tvg_h20_ab152_random_blue.sh"
)

echo "================================================================"
echo "ab152 第二轮消融队列启动: $(date)"
echo "共 ${#RUNS[@]} 个run, 每个 max_steps=125, save_steps=125(只存最终ckpt)"
echo "================================================================"

for script in "${RUNS[@]}"; do
    echo ""
    echo "================================================================"
    echo "[$(date)] 启动: $script"
    echo "================================================================"
    bash "${SCRIPT_DIR}/${script}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[WARN] $script 异常退出(rc=$rc), 继续下一个"
    else
        echo "[$(date)] $script 完成"
    fi
    # 等待进程完全退出, 避免下一个run的pgrep守卫误伤
    sleep 30
done

echo ""
echo "================================================================"
echo "全部消融run结束: $(date)"
echo "下一步: 对每个run评测 ckpt-125"
echo "================================================================"
