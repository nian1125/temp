#!/usr/bin/env bash
# 串行启动: 1) 纯GRPO(Run C) → 2) 纯OPSD-JSD(Run D)
# 自动处理: 训练前降级包+打nvtx补丁, 训练后还原
# 用法: nohup bash scripts/run_h20_serial_rl_opsd.sh >> /dockerdata/zhenjunchen/logs/serial_rl_opsd.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +%Y%m%d-%H%M)
LOG_DIR="/dockerdata/zhenjunchen/logs"
mkdir -p "${LOG_DIR}"

CUDA_ACC="/usr/local/lib/python3.12/dist-packages/deepspeed/accelerator/cuda_accelerator.py"

# ============================================================
# 退出时还原环境 (不管成功/失败/中断 都会执行)
# ============================================================
restore_base_env() {
    echo ""
    echo "============================================"
    echo ">>> Restoring base environment to original versions"
    echo "============================================"

    # 1. 还原 transformers → 5.3.0
    echo "  Restoring transformers==5.3.0 ..."
    pip install -i https://mirrors.cloud.tencent.com/pypi/simple transformers==5.3.0 2>&1 | tail -2

    # 2. 还原 trl → 0.27.0
    echo "  Restoring trl==0.27.0 ..."
    pip install -i https://mirrors.cloud.tencent.com/pypi/simple trl==0.27.0 2>&1 | tail -2

    # 3. 还原 deepspeed nvtx 补丁
    if [ -f "${CUDA_ACC}.bak" ]; then
        echo "  Restoring ${CUDA_ACC} from .bak ..."
        cp "${CUDA_ACC}.bak" "$CUDA_ACC"
        rm -f "${CUDA_ACC}.bak"
    fi

    # 4. 验证
    python3 -c "
import transformers, trl
t_ver = transformers.__version__
tr_ver = trl.__version__
print(f'  Verified: transformers={t_ver}, trl={tr_ver}')
assert t_ver == '5.3.0', f'Expected transformers 5.3.0, got {t_ver}'
assert tr_ver == '0.27.0', f'Expected trl 0.27.0, got {tr_ver}'
print('  Environment restored successfully.')
" || echo "  WARNING: Version check failed, please verify manually!"

    echo "============================================"
    echo "Restore finished: $(date)"
    echo "============================================"
}
trap restore_base_env EXIT

# ============================================================
# 训练前: 配置环境
# ============================================================
echo "============================================"
echo "Pre-training environment setup: $(date)"
echo "============================================"

# VENV
source /dockerdata/zhenjunchen/venv/bin/activate

# 1. 检查并备份 nvtx
echo ">>> Checking deepspeed nvtx patch..."
if [ ! -f "${CUDA_ACC}.bak" ]; then
    echo "  Backing up ${CUDA_ACC} ..."
    cp "$CUDA_ACC" "${CUDA_ACC}.bak"
    echo "  Applying nvtx compat fix ..."
    sed -i '/def range_push(self, msg, domain=None, category=None):/a\        return -1  # nvtx compat' "$CUDA_ACC"
else
    echo "  nvtx patch already applied (skip)."
fi

# 2. 降级 transformers + trl
echo ">>> Downgrading packages for training compatibility..."
pip install -i https://mirrors.cloud.tencent.com/pypi/simple transformers==4.57.1 2>&1 | tail -2
pip install -i https://mirrors.cloud.tencent.com/pypi/simple --no-deps trl==0.19.1 2>&1 | tail -2

# 3. 验证
echo ">>> Verifying package versions..."
python3 -c "
import transformers, trl
t_ver = transformers.__version__
tr_ver = trl.__version__
print(f'  transformers={t_ver} (expected 4.57.1)')
print(f'  trl={tr_ver} (expected 0.19.1)')
assert t_ver == '4.57.1', f'Expected transformers 4.57.1, got {t_ver}'
assert tr_ver == '0.19.1', f'Expected trl 0.19.1, got {tr_ver}'
print('  Package versions correct.')
"

echo "============================================"
echo "Environment ready. Starting training..."
echo "============================================"

# ============================================================
# Step 1: Pure GRPO (Run C)  kl=0, grpo=1
# ============================================================
LOG1="${LOG_DIR}/run_c_rl_only_${TIMESTAMP}.log"
echo ""
echo ">>> [1/2] Starting pure GRPO (Run C)"
echo "  log: ${LOG1}"
echo "  start: $(date)"

bash "${SCRIPT_DIR}/run_opd_tvg_h20_run_c_rl_only.sh" 2>&1 | tee "${LOG1}"
RC1=$?
echo "<<< [1/2] Pure GRPO finished, exit=${RC1}, $(date)"

if [ $RC1 -ne 0 ]; then
    echo "ERROR: Run C failed with code ${RC1}"
    exit $RC1
fi

# ============================================================
# Step 2: Pure OPSD-JSD (Run D)  kl=1, grpo=0
# ============================================================
LOG2="${LOG_DIR}/run_d_opsd_jsd_${TIMESTAMP}.log"
echo ""
echo ">>> [2/2] Starting pure OPSD-JSD (Run D)"
echo "  log: ${LOG2}"
echo "  start: $(date)"

bash "${SCRIPT_DIR}/run_opd_tvg_h20_run_d_opsd_only.sh" 2>&1 | tee "${LOG2}"
RC2=$?
echo "<<< [2/2] Pure OPSD-JSD finished, exit=${RC2}, $(date)"

echo ""
echo "============================================"
echo "All training done: $(date)"
echo "Logs:"
echo "  ${LOG1}"
echo "  ${LOG2}"
echo "============================================"

# 退出时 trap EXIT 会自动调用 restore_base_env
