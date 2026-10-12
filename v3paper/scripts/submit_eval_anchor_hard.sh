#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
OPD=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo
DIR=$(ls -dt $OPD/opd-*-anchor-hard-2026* 2>/dev/null | head -1)
[ -z "$DIR" ] && { echo "没找到anchor-hard训练目录"; exit 1; }
echo "评测: $DIR"; RES=$OPD/eval_anchor_hard_results.txt; echo "# anchor-hard ($(date)) $DIR" > $RES
for step in 350 300 250 200 150 100 50; do
  ckpt=$DIR/checkpoint-$step; [ -d "$ckpt" ] || { echo "[skip $step]"; continue; }
  OUT=/tmp/anchorhardev_$step; rm -f ${OUT}_*.jsonl
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i python scripts/think_eval.py --anchor --model_path "$ckpt" --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/charades_bench_gap.jsonl" --video_dir "/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens" --output "$OUT" --max_samples 100 --chunk 8 --index $i &
  done; wait
  m=$(python3 -c "import json,glob; xs=[json.loads(l)['tiou'] for f in glob.glob('${OUT}_*.jsonl') for l in open(f) if l.strip() and 'tiou' in l]; print(f'{sum(xs)/len(xs)*100:.2f} (n={len(xs)})' if xs else 'NA')")
  echo "step$step mIoU=$m" | tee -a $RES
done
cat $RES
