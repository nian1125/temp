#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把canonical Charades-TimeLens bench(dict{vid:{duration,spans,queries}})展平成
think_eval.py能读的全量jsonl(每行一个 {video_path,query,span,duration})。
这样 softrange/anchor 也能在【完整3363样本】上测, 和54.99/55.2同口径。

用法(primus登录节点, 纯CPU):
  python3 make_charades_bench_full.py \
    --bench /primus_xpfs_root/zhangyajie/czy/timelens/TimeLens/data/TimeLens-Bench/charades-timelens.json \
    --out   /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/charades_bench_full.jsonl
video_dir(think_eval用) = .../TimeLens/data/TimeLens-Bench/videos/charades , video_path = <vid>.mp4
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.load(open(args.bench))
    items = d.items() if isinstance(d, dict) else [(x.get("video", x.get("vid", "")), x) for x in d]
    n = 0
    with open(args.out, "w") as f:
        for vid, info in items:
            qs = info["queries"]; sps = info["spans"]; dur = info.get("duration", 0)
            for q, sp in zip(qs, sps):
                # sp 可能是 [s,e] 或 [[s,e]]
                if sp and isinstance(sp[0], (list, tuple)):
                    sp = sp[0]
                f.write(json.dumps({
                    "video_path": f"{vid}.mp4",
                    "query": q,
                    "span": [float(sp[0]), float(sp[1])],
                    "duration": float(dur),
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"✅ 展平 {n} 条 → {args.out}  (应≈3363)")


if __name__ == "__main__":
    main()
