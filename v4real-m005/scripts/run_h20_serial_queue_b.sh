#!/usr/bin/env bash
# 机器B串行队列: 等待当前训练结束 → replica152(Run G复刻) → ema_0995 → frozen
# 用法: nohup bash scripts/run_h20_serial_queue_b.sh >> /dockerdata/zhenjunchen/logs/serial_queue_b.log 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
TIMESTAMP=$(date +%Y%m%d-%H%M)
LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p "${LOG_DIR}"

VENV=/dockerdata/zhenjunchen/venv
[ -f "${VENV}/pyvenv.cfg" ] || { echo "FATAL: venv不存在"; exit 1; }
source "${VENV}/bin/activate"

# 等待当前训练结束 (每60s检查一次)
if pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; then
    echo ">>> 等待当前训练结束: $(date)"
    while pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; do sleep 60; done
    echo ">>> 当前训练已结束: $(date)"
fi

echo "============================================"
echo "串行队列启动: $(date)"
echo "  1) red_only   (第5组: 纯红框)"
echo "  2) replica    (Run G配置复刻: 纯红框+legacy+cached)"
echo "  3) ema_0995   (第3组: EMA teacher)"
echo "  4) frozen     (第3组: 冻结teacher)"
echo "数据: red_only/ema/frozen=charades393, replica=152原始; 全部师生同prompt"
echo "============================================"

declare -A RESULTS

run_one() {
    local tag="$1"
    local script="$2"
    local log="${LOG_DIR}/${tag}_${TIMESTAMP}.log"
    echo ""
    echo ">>> [${tag}] start: $(date)"
    echo "    log: ${log}"
    bash "${SCRIPT_DIR}/${script}" 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    echo "<<< [${tag}] finished, exit=${rc}, $(date)"
    RESULTS[$tag]=$rc
    return 0
}

run_one "red_only"    "run_opd_tvg_h20_ablation_red_only.sh"
run_one "replica"     "run_opd_tvg_h20_replica152_rung.sh"
run_one "ema_0995"    "run_opd_tvg_h20_ablation_ema_0995.sh"
run_one "frozen"      "run_opd_tvg_h20_ablation_frozen.sh"

echo ""
echo "============================================"
echo "队列完成: $(date)"
for tag in red_only replica ema_0995 frozen; do
    echo "  ${tag}: exit=${RESULTS[$tag]:-not_run}"
done
echo "============================================"
