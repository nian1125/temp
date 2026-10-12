#!/usr/bin/env bash
# H20 新机器 venv 初始化 (真venv方案, base零改动)
# 原理: python3 -m venv --system-site-packages
#   - 能用 base 的 torch/flash_attn/cv2 等 (只读引用)
#   - pip 安装/降级全部落在 venv 内, 不碰 base
#   - 交接时 rm -rf /dockerdata/zhenjunchen/venv 即完全干净
# 用法: bash scripts/setup_h20_venv.sh
set -euo pipefail

VENV=/dockerdata/zhenjunchen/venv
MIRROR=https://mirrors.cloud.tencent.com/pypi/simple

echo "============================================"
echo "H20 venv 初始化 (真venv, base零改动): $(date)"
echo "============================================"

# 0. 清理旧的假venv (如有)
if [ -f "${VENV}/bin/activate" ] && [ ! -f "${VENV}/pyvenv.cfg" ]; then
    echo ">>> 检测到旧版假venv, 清除重建 ..."
    rm -rf "${VENV}"
fi

# 1. 视频解码用 decord (qwen-vl-utils优先选decord后端), 训练路径不import cv2
#    libGL/cv2 无需检查, base 完全不动

# 2. 创建真venv (引用base包, 但pip安装只进venv)
if [ ! -f "${VENV}/pyvenv.cfg" ]; then
    echo ">>> 创建 venv (--system-site-packages) ..."
    python3.12 -m venv --system-site-packages "${VENV}" 2>/dev/null || {
        # ensurepip缺失且机器只有腾讯云内网源(apt不通) -> --without-pip, pip模块复用base
        echo "ensurepip缺失, 改用 --without-pip (pip复用base, 安装目标仍在venv) ..."
        rm -rf "${VENV}"
        python3.12 -m venv --system-site-packages --without-pip "${VENV}"
    }
else
    echo ">>> venv 已存在 (skip创建)"
fi

source "${VENV}/bin/activate"

# pip必须可用(--without-pip时复用base的pip模块) 且安装目标是venv
python3 -m pip --version >/dev/null 2>&1 || { echo "FATAL: pip不可用"; exit 1; }
PIP_TARGET=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo ">>> pip 安装目标: ${PIP_TARGET}"
case "${PIP_TARGET}" in
    ${VENV}/*) echo "    ✅ 在venv内, base安全" ;;
    *) echo "    ❌ 指向base! 中止"; exit 1 ;;
esac

# 3. venv内安装全部依赖 (transformers/trl降级也在这里, 不碰base)
# 注意: 不能加 --ignore-installed! 否则pip会把最新torch拖进venv覆盖base的2.10+cu129引用
# 前提: base里没有deepspeed (有的话pip会跳过导致import落到base)
echo ">>> 安装训练依赖到venv ..."
python3 -m pip install -i ${MIRROR} -q deepspeed==0.19.2 2>&1 | tail -1
python3 -m pip install -i ${MIRROR} -q transformers==4.57.1 2>&1 | tail -1
python3 -m pip install -i ${MIRROR} -q --no-deps trl==0.19.1 2>&1 | tail -1
python3 -m pip install -i ${MIRROR} -q decord imageio imageio-ffmpeg qwen-vl-utils nncore opencv-python-headless weave 2>&1 | tail -1

# 3.5 关键断言: torch必须来自base (2.10.0+cu129), 绝不允许venv里有torch
python3 -c "
import torch
assert not torch.__file__.startswith('${VENV}'), f'torch被装进了venv! {torch.__file__} -> 会脱离base的cu129'
print(f'  torch={torch.__version__} (来自base, ✅)')
assert torch.__version__.startswith('2.10'), torch.__version__
"

# 4. nvtx补丁 (打在venv的deepspeed上)
# 注意: import deepspeed 会打印warning, 只取最后一行作为路径
CUDA_ACC=$(python3 -c "import deepspeed, os; print(os.path.join(os.path.dirname(deepspeed.__file__), 'accelerator/cuda_accelerator.py'))" 2>/dev/null | tail -1)
echo ">>> nvtx补丁目标: ${CUDA_ACC}"
case "${CUDA_ACC}" in
    ${VENV}/*) : ;;
    *) echo "    ❌ deepspeed在base里! 中止"; exit 1 ;;
esac
if [ ! -f "${CUDA_ACC}.bak" ]; then
    cp "${CUDA_ACC}" "${CUDA_ACC}.bak"
    sed -i '/def range_push(self, msg, domain=None, category=None):/a\        return -1  # nvtx compat' "${CUDA_ACC}"
    echo "    nvtx补丁已打 (venv内)"
else
    echo "    nvtx补丁已存在 (skip)"
fi

# 5. 验证 (venv内版本 + 确认base未被改动)
echo ">>> 验证 venv 内版本 ..."
python3 -c "
import transformers, trl, torch, deepspeed, decord
assert transformers.__version__ == '4.57.1', transformers.__version__
assert trl.__version__ == '0.19.1', trl.__version__
assert transformers.__file__.startswith('${VENV}'), transformers.__file__
assert trl.__file__.startswith('${VENV}'), trl.__file__
assert deepspeed.__file__.startswith('${VENV}'), deepspeed.__file__
print(f'  venv: transformers={transformers.__version__}, trl={trl.__version__}, deepspeed={deepspeed.__version__}')
print(f'  视频后端: decord={decord.__version__} (训练实际使用)')
print(f'  base引用: torch={torch.__version__}')
try:
    import flash_attn; print(f'  base引用: flash_attn={flash_attn.__version__}')
except Exception as e:
    print(f'  flash_attn: {e}')
try:
    import cv2; print(f'  cv2={cv2.__version__} (训练不用, 仅备用)')
except ImportError as e:
    print(f'  cv2 不可用 ({e}) —— 训练路径不import cv2, 可忽略')
"

echo ">>> 验证 base 未被降级 ..."
deactivate
/usr/bin/python3.12 -c "
import transformers, trl
print(f'  base: transformers={transformers.__version__}, trl={trl.__version__}')
assert transformers.__version__ == '5.3.0', 'base transformers 被改了!'
assert trl.__version__ == '0.27.0', 'base trl 被改了!'
print('  ✅ base 保持原版')
" 2>&1 || echo "  ⚠️ base版本不是 5.3.0/0.27.0, 请检查 (可能今天早些时候的假venv污染)"

echo "============================================"
echo "venv 初始化完成: ${VENV}"
echo "交接清理: rm -rf ${VENV}"
echo "============================================"
