#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
ROOT=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
DIR=$(ls -dt $ROOT/output/opd-grpo/opd-*-anchor-sft-2026* 2>/dev/null | head -1)
[ -z "$DIR" ] && { echo "❌ 没找到 anchor-sft 训练目录"; exit 1; }
echo "评测: $DIR"
RES=$ROOT/output/opd-grpo/eval_anchor_sft_full_results.txt
echo "# anchor-sft FULL(3363) ($(date)) $DIR" > $RES
for step in ${STEPS:-200 150 100 250 300 350 50}; do
  ckpt=$DIR/checkpoint-$step; [ -d "$ckpt" ] || { echo "[skip $step]"; continue; }
  OUT=/tmp/anchorsftfull_$step; rm -f ${OUT}_*.jsonl
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i python scripts/think_eval.py --anchor \
      --model_path "$ckpt" --anno_path "/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/charades_bench_full.jsonl" --video_dir "/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens/data/TimeLens-Bench/videos/charades" --output "$OUT" --chunk 8 --index $i &
  done; wait
  python3 - "$OUT" "$step" "$RES" <<'PYEOF'
import json,glob,sys
out,step,res=sys.argv[1],sys.argv[2],sys.argv[3]
xs=[json.loads(l)["tiou"] for f in glob.glob(out+"_*.jsonl") for l in open(f) if l.strip() and '"tiou"' in l]
n=len(xs) or 1
line=f"step{step}: R0.3={sum(t>=0.3 for t in xs)/n*100:.2f} R0.5={sum(t>=0.5 for t in xs)/n*100:.2f} R0.7={sum(t>=0.7 for t in xs)/n*100:.2f} mIoU={sum(xs)/n*100:.2f} (n={len(xs)})"
print(line); open(res,"a").write(line+"\n")
PYEOF
done
cat $RES
