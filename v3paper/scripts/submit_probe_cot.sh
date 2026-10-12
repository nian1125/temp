#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 探针: SFT模型能不能产grounding CoT? 每样本跑[裸答案]+[先思考再答], 记录全文+tIoU。Charades-Bench, 实时写+可续。
MODEL_PATH=/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT
ANNO=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/charades_bench_gap.jsonl
VIDEO_DIR=/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
MAXS="${MAXS:-50}"
OUTDIR=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/probe_cot
mkdir -p "$OUTDIR"
OUT="$OUTDIR/p"

echo "[CFG] probe_cot SFT基座 anno=Charades-Bench maxs=$MAXS (总≈$((MAXS*8)))"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python scripts/probe_cot.py \
    --model_path "$MODEL_PATH" --anno_path "$ANNO" --video_dir "$VIDEO_DIR" \
    --output "$OUT" --max_samples "$MAXS" --chunk 8 --index $i &
done
wait

echo "=========== 探针汇总 ==========="
python3 scripts/scan_probe_cot.py "$OUTDIR"
echo "[DONE probe_cot]"
