#!/usr/bin/env bash
# 串行启动消融: 1) cbm_v1 (baseline蓝框自身) → 2) cbm_v2 (baseline蓝框100K) → 3) fixed_baseline (第7组修复版)
# 前置: 已运行 setup_h20_venv.sh (真venv, 依赖/降级/nvtx补丁都在venv内, 不碰base)
# 单个run失败不阻塞后续run, 最后汇总exit code
# 用法: nohup bash scripts/run_h20_serial_ablation_g1_g7.sh >> /dockerdata/zhenjunchen/logs/serial_ablation_g1_g7.log 2>&1 &
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
TIMESTAMP=$(date +%Y%m%d-%H%M)
# 日志写到ceph, 开发侧(apdcephfs)可直接查看/debug
LOG_DIR="${PROJECT_ROOT}/output/logs"
mkdir -p "${LOG_DIR}"

VENV=/dockerdata/zhenjunchen/venv

# venv 检查 (真venv必须有pyvenv.cfg)
if [ ! -f "${VENV}/pyvenv.cfg" ]; then
    echo "FATAL: ${VENV} 不存在或是旧版假venv!"
    echo "请先运行: bash ${SCRIPT_DIR}/setup_h20_venv.sh"
    exit 1
fi
source "${VENV}/bin/activate"

# 版本断言 (全部来自venv, base不动)
python3 -c "
import transformers, trl, deepspeed
assert transformers.__version__ == '4.57.1', transformers.__version__
assert trl.__version__ == '0.19.1', trl.__version__
assert transformers.__file__.startswith('${VENV}'), 'transformers 不在venv内!'
assert deepspeed.__file__.startswith('${VENV}'), 'deepspeed 不在venv内!'
print(f'venv OK: transformers={transformers.__version__}, trl={trl.__version__}, deepspeed={deepspeed.__version__}')
"
if [ $? -ne 0 ]; then
    echo "FATAL: venv版本校验失败, 请先运行 setup_h20_venv.sh"
    exit 1
fi

echo "============================================"
echo "Starting 3 serial ablation runs: $(date)"
echo "  1) cbm_v1         (coarse_box_multi, 蓝框=393自身)"
echo "  2) cbm_v2         (coarse_box_multi, 蓝框=100K)"
echo "  3) fixed_baseline (第7组: drifting+zoom_expand修复)"
echo "============================================"

declare -A RESULTS

run_one() {
    local tag="$1"
    local script="$2"
    local log="${LOG_DIR}/${tag}_${TIMESTAMP}.log"
    echo ""
    echo ">>> [${tag}] start: $(date)"
    echo "    script: ${script}"
    echo "    log:    ${log}"
    bash "${SCRIPT_DIR}/${script}" 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    echo "<<< [${tag}] finished, exit=${rc}, $(date)"
    RESULTS[$tag]=$rc
    return 0
}

run_one "cbm_v1" "run_opd_tvg_h20_ablation_cbm_v1.sh"
run_one "cbm_v2" "run_opd_tvg_h20_ablation_cbm_v2.sh"
run_one "fixed_baseline" "run_opd_tvg_h20_ablation_fixed_baseline.sh"

echo ""
echo "============================================"
echo "All 3 runs done: $(date)"
echo "Results:"
for tag in cbm_v1 cbm_v2 fixed_baseline; do
    echo "  ${tag}: exit=${RESULTS[$tag]:-not_run}"
done
echo "Logs in: ${LOG_DIR}/*_${TIMESTAMP}.log"
echo "============================================"

for tag in cbm_v1 cbm_v2 fixed_baseline; do
    if [ "${RESULTS[$tag]:-1}" -ne 0 ]; then
        exit 1
    fi
done
exit 0
