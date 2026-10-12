"""NPU (昇腾) 兼容补丁: 把 torch.cuda API 映射到 torch.npu。

用法: 在入口脚本最前面
    from zwz.npu_patch import patch_npu
    patch_npu()

原理: torch_npu 已注册 npu 设备, 但训练代码里散落的 torch.cuda.empty_cache()/
      torch.cuda.current_device() 等调用在 NPU 下会报错。这里统一做别名映射。
"""

import torch

_patched = False


def patch_npu(force=False):
    global _patched
    if _patched and not force:
        return
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("[npu_patch] torch_npu 未安装, 跳过补丁 (假设是 CUDA 环境)")
        return

    _patched = True
    print("[npu_patch] 检测到 torch_npu, 映射 torch.cuda → torch.npu")

    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = torch.npu.device_count
    torch.cuda.current_device = torch.npu.current_device
    torch.cuda.set_device = torch.npu.set_device
    torch.cuda.empty_cache = torch.npu.empty_cache
    torch.cuda.synchronize = torch.npu.synchronize
    torch.cuda.memory_allocated = lambda *a, **k: torch.npu.memory_allocated(*a, **k)
    torch.cuda.memory_reserved = lambda *a, **k: torch.npu.memory_reserved(*a, **k)
    torch.cuda.max_memory_allocated = lambda *a, **k: torch.npu.max_memory_allocated(*a, **k)
    torch.cuda.get_device_name = lambda *a, **k: "Ascend910B2C"
    torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
