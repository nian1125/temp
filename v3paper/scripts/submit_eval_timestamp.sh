
# 切换到指定目录
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3

#!/bin/bash

pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
python --version
pip show flash-attn
pip uninstall -y flash-attn
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

python -c "import flash_attn; print('✅ Success!')"

free -h
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# 先下载最新代码
ossutil cp oss://quark-llm/primus/datasets/qianzi.qz/czy_transfer/all_updates_20260624_v2.tar.gz /tmp/all_updates_20260624_v2.tar.gz -c /primus_xpfs_root/zhangyajie/czy/aaacode/ossutilconfig.txt
tar xzf /tmp/all_updates_20260624_v2.tar.gz

# A1: timestamp 策略评测 (~2h)
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/run_eval_teacher.sh timestamp

# A2: gt_text+timestamp 策略评测 (~2h)
bash /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/scripts/run_eval_teacher.sh gt_text+timestamp
