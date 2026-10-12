#!/usr/bin/env bash
# H20 Qwen3.5 专用 venv (独立环境, 不影响现有 Qwen3-VL 训练的 venv)
# 与 setup_h20_venv.sh 唯一区别: transformers 4.57.1 -> 5.13.0 (qwen3_5 支持)
# 原理: python3 -m venv --system-site-packages
#   - torch/flash_attn 等复用 base (只读引用)
#   - transformers 5.13.0 + trl 0.19.1 + deepspeed 0.19.2 装进本 venv
#   - 交接清理: rm -rf /dockerdata/zhenjunchen/venv-qwen35
# 用法: bash scripts/setup_h20_venv_qwen35.sh
set -euo pipefail

VENV=/dockerdata/zhenjunchen/venv-qwen35
MIRROR=https://mirrors.cloud.tencent.com/pypi/simple

echo "============================================"
echo "H20 Qwen3.5 venv 初始化 (独立venv, base零改动): $(date)"
echo "  VENV=${VENV}"
echo "============================================"

# 1. 创建真venv (引用base包, 但pip安装只进venv)
if [ ! -f "${VENV}/pyvenv.cfg" ]; then
    echo ">>> 创建 venv (--system-site-packages) ..."
    python3.12 -m venv --system-site-packages "${VENV}" 2>/dev/null || {
        echo "ensurepip缺失, 改用 --without-pip (pip复用base, 安装目标仍在venv) ..."
        rm -rf "${VENV}"
        python3.12 -m venv --system-site-packages --without-pip "${VENV}"
    }
else
    echo ">>> venv 已存在 (skip创建)"
fi

source "${VENV}/bin/activate"

# pip必须可用 且安装目标是本venv
python3 -m pip --version >/dev/null 2>&1 || { echo "FATAL: pip不可用"; exit 1; }
PIP_TARGET=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo ">>> pip 安装目标: ${PIP_TARGET}"
case "${PIP_TARGET}" in
    ${VENV}/*) echo "    ✅ 在venv-qwen35内, base和旧venv都安全" ;;
    *) echo "    ❌ 指向别处! 中止"; exit 1 ;;
esac

# 2. 安装依赖 (注意: 不能加 --ignore-installed! 会把最新torch拖进venv覆盖base引用)
echo ">>> 安装训练依赖到 venv-qwen35 ..."
python3 -m pip install -i ${MIRROR} -q deepspeed==0.19.2 2>&1 | tail -1
python3 -m pip install -i ${MIRROR} -q transformers==5.13.0 2>&1 | tail -1
python3 -m pip install -i ${MIRROR} -q --no-deps trl==0.19.1 2>&1 | tail -1
python3 -m pip install -i ${MIRROR} -q decord imageio imageio-ffmpeg qwen-vl-utils nncore opencv-python-headless weave 2>&1 | tail -1

# 2.5 (可选, best-effort) GatedDeltaNet 快速路径: flash-linear-attention
# 不装也能跑 (transformers回退到torch实现, 慢一些); 装不上不阻塞
echo ">>> (可选) 安装 flash-linear-attention (GatedDeltaNet 加速) ..."
python3 -m pip install -i ${MIRROR} -q flash-linear-attention 2>&1 | tail -1 || \
    echo "    ⚠️ flash-linear-attention 安装失败, 将用torch回退实现(可训练, 速度略慢)"

# 3. 关键断言: torch必须来自base (2.10.0+cu129), 绝不允许venv里有torch
python3 -c "
import torch
assert not torch.__file__.startswith('${VENV}'), f'torch被装进了venv! {torch.__file__} -> 会脱离base的cu129'
print(f'  torch={torch.__version__} (来自base, ✅)')
assert torch.__version__.startswith('2.10'), torch.__version__
"

# 4. nvtx补丁 (打在本venv的deepspeed上)
CUDA_ACC=$(python3 -c "import deepspeed, os; print(os.path.join(os.path.dirname(deepspeed.__file__), 'accelerator/cuda_accelerator.py'))" 2>/dev/null | tail -1)
echo ">>> nvtx补丁目标: ${CUDA_ACC}"
case "${CUDA_ACC}" in
    ${VENV}/*) : ;;
    *) echo "    ❌ deepspeed在base里! 中止"; exit 1 ;;
esac
if [ ! -f "${CUDA_ACC}.bak" ]; then
    cp "${CUDA_ACC}" "${CUDA_ACC}.bak"
    sed -i '/def range_push(self, msg, domain=None, category=None):/a\        return -1  # nvtx compat' "${CUDA_ACC}"
    echo "    nvtx补丁已打 (venv-qwen35内)"
else
    echo "    nvtx补丁已存在 (skip)"
fi

# 5. 验证
echo ">>> 验证 venv-qwen35 内版本 ..."
python3 -c "
import transformers, trl, torch, deepspeed, decord
assert transformers.__version__ == '5.13.0', transformers.__version__
assert trl.__version__ == '0.19.1', trl.__version__
assert transformers.__file__.startswith('${VENV}'), transformers.__file__
assert trl.__file__.startswith('${VENV}'), trl.__file__
assert deepspeed.__file__.startswith('${VENV}'), deepspeed.__file__
print(f'  venv-qwen35: transformers={transformers.__version__}, trl={trl.__version__}, deepspeed={deepspeed.__version__}')
print(f'  base引用: torch={torch.__version__}')
try:
    import flash_attn; print(f'  base引用: flash_attn={flash_attn.__version__}')
except Exception as e:
    print(f'  flash_attn: {e}')
try:
    import fla; print(f'  flash-linear-attention: ✅ (GatedDeltaNet快速路径)')
except ImportError:
    print(f'  flash-linear-attention: 未安装, GatedDeltaNet用torch回退(可训练,略慢)')
"

# 5.5 验证 qwen3_5 支持 + Qwen3.5-2B 配置可加载
echo ">>> 验证 qwen3_5 模型支持 ..."
python3 -c "
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
assert 'qwen3_5' in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, 'transformers无qwen3_5映射!'
print(f'  qwen3_5 -> {MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES[\"qwen3_5\"]}')
from transformers import AutoConfig, AutoProcessor
cfg = AutoConfig.from_pretrained('/mnt/250T_ceph/zhenjunchen/models/Qwen3.5-2B')
print(f'  Qwen3.5-2B config: model_type={cfg.model_type}, arch={cfg.architectures}')
proc = AutoProcessor.from_pretrained('/mnt/250T_ceph/zhenjunchen/models/Qwen3.5-2B')
print(f'  processor: {type(proc).__name__} ✅')
"

echo ">>> 验证 base 未被改动 ..."
deactivate
/usr/bin/python3.12 -c "
import transformers, trl
print(f'  base: transformers={transformers.__version__}, trl={trl.__version__}')
assert transformers.__version__ == '5.3.0', 'base transformers 被改了!'
assert trl.__version__ == '0.27.0', 'base trl 被改了!'
print('  ✅ base 保持原版')
" 2>&1 || echo "  ⚠️ base版本异常, 请检查"

echo "============================================"
echo "venv-qwen35 初始化完成: ${VENV}"
echo "交接清理: rm -rf ${VENV}"
echo "============================================"
