#!/usr/bin/env bash
# 串行: Run G (TimeLens SFT) → Run H (TimeLens2-8B-SFT)
# 共用一次环境降级，最后统一还原
# 用法: nohup bash scripts/run_h20_serial_reviewed.sh >> /dockerdata/zhenjunchen/logs/serial_reviewed.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CUDA_ACC="/usr/local/lib/python3.12/dist-packages/deepspeed/accelerator/cuda_accelerator.py"

restore_base_env() {
    echo ">>> Restoring base environment"
    pip install -i https://mirrors.cloud.tencent.com/pypi/simple transformers==5.3.0 2>&1 | tail -2
    pip install -i https://mirrors.cloud.tencent.com/pypi/simple trl==0.27.0 2>&1 | tail -2
    [ -f "${CUDA_ACC}.bak" ] && cp "${CUDA_ACC}.bak" "$CUDA_ACC" && rm -f "${CUDA_ACC}.bak"
    echo ">>> Restore done: $(date)"
}
trap restore_base_env EXIT

echo "============================================"
echo "Serial: Run G → Run H, start: $(date)"
echo "============================================"

source /dockerdata/zhenjunchen/venv/bin/activate

echo ">>> Downgrading packages (once for both runs)"
pip install -i https://mirrors.cloud.tencent.com/pypi/simple transformers==4.57.1 2>&1 | tail -2
pip install -i https://mirrors.cloud.tencent.com/pypi/simple --no-deps trl==0.19.1 2>&1 | tail -2
[ ! -f "${CUDA_ACC}.bak" ] && cp "$CUDA_ACC" "${CUDA_ACC}.bak"
sed -i '/def range_push(self, msg, domain=None, category=None):/a\        return -1  # nvtx compat' "$CUDA_ACC"

echo ""
echo ">>> [1/2] Run G: TimeLens SFT base"
SKIP_SETUP=1 NO_RESTORE=1 bash "${SCRIPT_DIR}/run_opd_tvg_h20_run_g_reviewed.sh" 2>&1 | sed 's/^/[G] /'
echo "<<< Run G done: $(date)"

echo ""
echo ">>> [2/2] Run H: TimeLens2-8B-SFT base"
SKIP_SETUP=1 NO_RESTORE=1 bash "${SCRIPT_DIR}/run_opd_tvg_h20_run_h_timelens2.sh" 2>&1 | sed 's/^/[H] /'
echo "<<< Run H done: $(date)"

echo ""
echo "============================================"
echo "All done: $(date)"
echo "============================================"
