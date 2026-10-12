#!/bin/bash
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# 特权predictor on Charades-Bench(真测试集, 非易训练子集): SFT基座, 一次跑全部10特权,
# 每样本记录plain+各策略tIoU+KL → 同时出[全bench]和[学生低×teacher高 难子集]两张表。

MODEL_PATH=/primus_xpfs_root/zhangyajie/ckpt/TimeLens-Qwen3-VL-8B-SFT
ANNO=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/charades_bench_gap.jsonl
VIDEO_DIR=/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
STRATEGIES="plain,numberit,coarse_box,coarse_box_hint,coarse_text,numberit+coarse_box,timestamp,timestamp+coarse_box,red_border,numberit+red_border,gt_text"
MAXS="${MAXS:-50}"   # 每卡条数; 8卡 → 总 8*MAXS ≈ 400 个样本
TAG="sft-bench"
OUTDIR="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/gap/${TAG}"
CACHE="/tmp/gapcache_${TAG}"
rm -rf "$OUTDIR" "$CACHE"; mkdir -p "$OUTDIR"
OUT="$OUTDIR/g"

echo "[CFG] model=$TAG anno=Charades-Bench strategies=$STRATEGIES maxs=$MAXS (总≈$((MAXS*8)))"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python scripts/measure_gap.py \
    --model_path "$MODEL_PATH" --anno_path "$ANNO" --video_dir "$VIDEO_DIR" \
    --strategies "$STRATEGIES" --output "$OUT" --cache_dir "$CACHE" \
    --image_patch_size 16 --max_samples "$MAXS" --chunk 8 --index $i &
done
wait

echo "=========== 汇总 [全Bench + 难子集] ==========="
python3 - "$OUTDIR" "$STRATEGIES" <<'PY'
import sys, json, glob
outdir, strats = sys.argv[1], [s.strip() for s in sys.argv[2].split(",")]
recs=[]
for f in glob.glob(f"{outdir}/g_*.jsonl"):
    for l in open(f):
        if l.strip(): recs.append(json.loads(l))

def table(subset, title):
    print(f"\n===== {title} (n={len(subset)}) =====")
    pl=[r['tiou']['plain'] for r in subset if r['tiou'].get('plain') is not None]
    pm=sum(pl)/len(pl)*100 if pl else 0
    print(f"{'strategy':<22}{'n':>5}{'mIoU':>8}{'Δ(有用)':>10}{'边界KL':>10}")
    print(f"{'plain':<22}{len(pl):>5}{pm:>8.2f}{'(基准)':>10}{'—':>10}")
    rows=[]
    for s in strats:
        if s=='plain': continue
        pairs=[(r['tiou']['plain'], r['tiou'][s]) for r in subset
               if r['tiou'].get('plain') is not None and r['tiou'].get(s) is not None]
        if not pairs: continue
        d=sum(q-p for p,q in pairs)/len(pairs)*100
        pr=sum(q for _,q in pairs)/len(pairs)*100
        kls=[r['kl'][s] for r in subset if r.get('kl',{}).get(s) is not None]
        kl=sum(kls)/len(kls) if kls else None
        rows.append((s,len(pairs),pr,d,kl))
    for s,n,pr,d,kl in sorted(rows,key=lambda x:-x[3]):
        klc=f"{kl:.4f}" if kl is not None else "n/a"
        print(f"{s:<22}{n:>5}{pr:>8.2f}{d:>+10.2f}{klc:>10}")

print(f"总记录 n={len(recs)}")
table(recs, "全 Charades-Bench")
for thr in (0.5, 0.3):
    hard=[r for r in recs if r['tiou'].get('plain') is not None and r['tiou']['plain'] < thr]
    table(hard, f"学生低×teacher高: 难子集 plain_tIoU<{thr}")
print("\n判据: Δ>0且KL小=理想特权。重点看难子集——特权在学生做不好处是否Δ转正(coarse硬墙是否被推翻)")
PY
echo "[DONE sft-bench]"
