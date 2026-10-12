#!/usr/bin/env bash
# ============================================================
# ab152 第三轮消融串行队列: 损失权重配比 (JSD:GRPO)
#   1. kl05  (0.5:1)
#   2. kl20  (2:1)
#   3. kl30  (3:1)
# 端点已有: 纯GRPO(1:0) 纯JSD(0:1) baseline(1:1), 补齐后共6个点成曲线
# 开头自动等待当前训练(random_blue)结束再启动, 单个失败不阻塞后续
# 用法: nohup bash scripts/run_h20_serial_ab152_round3.sh >> /dockerdata/zhenjunchen/logs/serial_ab152_round3.log 2>&1 &
# ============================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUNS=(
    "run_opd_tvg_h20_ab152_kl05.sh"
    "run_opd_tvg_h20_ab152_kl20.sh"
    "run_opd_tvg_h20_ab152_kl30.sh"
)

echo "================================================================"
echo "ab152 第三轮消融队列(权重配比): $(date)"
echo "================================================================"

# 等待当前训练进程结束(如 random_blue 还在跑)
while pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; do
    echo "[$(date)] 等待当前训练结束..."
    sleep 120
done

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
    sleep 30
done

echo ""
echo "================================================================"
echo "权重配比消融全部结束: $(date)"
echo "配比曲线6点: 1:0(pure-grpo) 0.5:1 1:1(baseline) 2:1 3:1 0:1(jsd-only)"
echo "================================================================"
