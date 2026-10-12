#!/bin/bash
# 单模型特权预测器。用法: MODEL_PATH=... STRATEGIES=... PATCH=16 MAXS=40 bash submit_measure_gap.sh
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

MODEL_PATH="${MODEL_PATH:-/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct}"
STRATEGIES="${STRATEGIES:-plain,numberit}"
PATCH="${PATCH:-16}"
MAXS="${MAXS:-60}"
ANNO="${ANNO:-/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens}"
TAG="${TAG:-$(basename "$MODEL_PATH")}"
OUTDIR="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/gap/${TAG}"
CACHE="/tmp/gapcache_${TAG}"

# 关键: 清掉旧输出 + 旧teacher缓存(否则复用padding坏视频 / glob到历史jsonl)
rm -rf "$OUTDIR" "$CACHE"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/g"

echo "[CFG] model=$TAG strategies=$STRATEGIES patch=$PATCH maxs=$MAXS"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python scripts/measure_gap.py \
    --model_path "$MODEL_PATH" --anno_path "$ANNO" --video_dir "$VIDEO_DIR" \
    --strategies "$STRATEGIES" --output "$OUT" --cache_dir "$CACHE" \
    --image_patch_size "$PATCH" --max_samples "$MAXS" --chunk 8 --index $i &
done
wait

echo "=========== 预测器汇总 [$TAG] ==========="
python3 - "$OUTDIR" "$STRATEGIES" <<'PY'
import sys, json, glob
outdir, strats = sys.argv[1], [s.strip() for s in sys.argv[2].split(",")]
recs = []
for f in glob.glob(f"{outdir}/g_*.jsonl"):
    for l in open(f):
        if l.strip(): recs.append(json.loads(l))
# 只在"plain和该策略都成功"的样本上算 Δ
def agg(s):
    pairs = [(r["tiou"].get("plain"), r["tiou"].get(s)) for r in recs
             if r["tiou"].get("plain") is not None and r["tiou"].get(s) is not None]
    if not pairs: return (0,0,0,None)
    pl = sum(p for p,_ in pairs)/len(pairs)*100
    pr = sum(q for _,q in pairs)/len(pairs)*100
    kls = [r["kl"][s] for r in recs if r.get("kl",{}).get(s) is not None]
    kl = sum(kls)/len(kls) if kls else None
    return (pr, len(pairs), pr-pl, kl)
print(f"总样本记录 n={len(recs)}")
print(f"{'strategy':<16}{'n':>5}{'mIoU':>8}{'Δ(有用)':>10}{'边界KL(可迁移)':>16}")
for s in strats:
    if s=="plain":
        pl=[r['tiou']['plain'] for r in recs if r['tiou'].get('plain') is not None]
        m=sum(pl)/len(pl)*100 if pl else 0
        print(f"{'plain':<16}{len(pl):>5}{m:>8.2f}{'  (基准)':>10}{'—':>16}"); continue
    pr,n,d,kl = agg(s)
    kls = f"{kl:.4f}" if kl is not None else "n/a"
    print(f"{s:<16}{n:>5}{pr:>8.2f}{d:>+10.2f}{kls:>16}")
print("判据: Δ>0 且 边界KL小 = 理想特权(纯蒸馏可直接用)")
PY
echo "[DONE $TAG]"
