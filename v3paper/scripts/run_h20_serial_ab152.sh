#!/usr/bin/env bash
# 152消融串行队列: no_rescue → pure_grpo → frozen → fkl → rkl (每个 max_steps=225)
# 用法: nohup bash scripts/run_h20_serial_ab152.sh >> /dockerdata/zhenjunchen/logs/serial_ab152.log 2>&1 &
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
echo "152消融串行队列启动: $(date)"
echo "  1) no_rescue  (dead_rescue=False)"
echo "  2) pure_grpo  (kl_weight=0, 蒸馏零梯度)"
echo "  3) frozen     (冻结teacher)"
echo "  4) fkl        (前向KL替代JSD)"
echo "  5) rkl        (反向KL替代JSD)"
echo "baseline: fixed-all-152 (eval峰值55.62@ckpt125), 全部 max_steps=225"
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

run_one "ab152_no_rescue" "run_opd_tvg_h20_ab152_no_rescue.sh"
run_one "ab152_pure_grpo" "run_opd_tvg_h20_ab152_pure_grpo.sh"
run_one "ab152_frozen"    "run_opd_tvg_h20_ab152_frozen.sh"
run_one "ab152_fkl"      "run_opd_tvg_h20_ab152_fkl.sh"
run_one "ab152_rkl"      "run_opd_tvg_h20_ab152_rkl.sh"

echo ""
echo "============================================"
echo "队列完成: $(date)"
for tag in ab152_no_rescue ab152_pure_grpo ab152_frozen ab152_fkl ab152_rkl; do
    echo "  ${tag}: exit=${RESULTS[$tag]:-not_run}"
done
echo "============================================"
