#!/bin/bash
# base Qwen3-VL-8B 在 charades-timelens bench 上推理, 预测传回 OSS 供挑 bad case。
set -e
TL=/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
cd "$TL"
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
export model_path=/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct
export datasets="charades-timelens"
export pred_path=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/badcase_preds
export min_tokens=64 total_tokens=14336 FPS=2
bash scripts/eval_timelens_bench.sh
# 上传聚合结果
OUT=$(ls -dt $pred_path/*/ 2>/dev/null | head -1)
F="${OUT}charades-timelens.jsonl"
OSS=oss://quark-llm/primus/datasets/qianzi.qz/czy_transfer/badcases/preds
name="BASE-Qwen3VL8B__charades-timelens.jsonl"
[ -f "$F" ] && ossutil cp -f "$F" "$OSS/$name" >/dev/null 2>&1 && echo "✅ 已传: $name" || echo "❌ 未找到 $F"
