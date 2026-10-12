#!/usr/bin/env bash
# 串行启动消融 第2+3组 (优先级排序):
#   1) pure_grpo      (kl=0, 纯强化, 无teacher forward, ~2.5h)
#   2) pure_opsd_jsd  (grpo=0, 纯蒸馏JSD, ~3h)
#   3) frozen         (冻结teacher, ~3h)
#   4) ema_0995       (EMA teacher, ~3h, 时间不够则部分完成, ckpt每25步落盘)
#   5) pure_opsd_fkl  (优先级低)
#   6) pure_opsd_rkl  (优先级低)
# 环境: 真venv, base零改动, 跑完无需还原
# 用法: nohup bash scripts/run_h20_serial_ablation_g2_g3.sh >> /dockerdata/zhenjunchen/logs/serial_ablation_g2_g3.log 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
TIMESTAMP=$(date +%Y%m%d-%H%M)
LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p "${LOG_DIR}"

VENV=/dockerdata/zhenjunchen/venv

if [ ! -f "${VENV}/pyvenv.cfg" ]; then
    echo "FATAL: ${VENV} 不存在! 请先运行: bash ${SCRIPT_DIR}/setup_h20_venv.sh"
    exit 1
fi
source "${VENV}/bin/activate"

python3 -c "
import transformers, trl, deepspeed
assert transformers.__version__ == '4.57.1'
assert trl.__version__ == '0.19.1'
assert transformers.__file__.startswith('${VENV}')
assert deepspeed.__file__.startswith('${VENV}')
print(f'venv OK: transformers={transformers.__version__}, trl={trl.__version__}, deepspeed={deepspeed.__version__}')
"
if [ $? -ne 0 ]; then
    echo "FATAL: venv版本校验失败, 请先运行 setup_h20_venv.sh"
    exit 1
fi

echo "============================================"
echo "Serial ablation G2+G3: $(date)"
echo "  1) pure_grpo   2) pure_opsd_jsd   3) frozen"
echo "  4) ema_0995    5) pure_opsd_fkl  6) pure_opsd_rkl"
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

run_one "pure_grpo"     "run_opd_tvg_h20_ablation_pure_grpo.sh"
run_one "pure_opsd_jsd" "run_opd_tvg_h20_ablation_pure_opsd_jsd.sh"
run_one "frozen"        "run_opd_tvg_h20_ablation_frozen.sh"
run_one "ema_0995"      "run_opd_tvg_h20_ablation_ema_0995.sh"
run_one "pure_opsd_fkl" "run_opd_tvg_h20_ablation_pure_opsd_fkl.sh"
run_one "pure_opsd_rkl" "run_opd_tvg_h20_ablation_pure_opsd_rkl.sh"

echo ""
echo "============================================"
echo "Serial done: $(date)"
echo "Results:"
for tag in pure_grpo pure_opsd_jsd frozen ema_0995 pure_opsd_fkl pure_opsd_rkl; do
    echo "  ${tag}: exit=${RESULTS[$tag]:-not_run}"
done
echo "Logs in: ${LOG_DIR}/*_${TIMESTAMP}.log"
echo ""
echo ">>> 交接清理: base 全程零改动, 无需还原"
echo ">>> 如需彻底清理: rm -rf /dockerdata/zhenjunchen/venv"
echo "============================================"
