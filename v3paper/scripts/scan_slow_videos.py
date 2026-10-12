#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描训练日志, 聚合每个视频的最慢 decord 解码耗时,
输出「病态视频」清单(源视频 train_videos), 供转码脚本使用.

用法:
    python3 scripts/scan_slow_videos.py [--threshold 30]

输出:
    output/slow_source_videos.txt   每行一个源视频 basename (需转码)
    output/slow_videos_report.txt   完整排名报告 (含 teacher_cache)
"""
import re, os, glob, argparse, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")

PAT = re.compile(r"video_path='([^']+)'.*?time=([0-9.]+)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=30.0,
                    help="最大解码耗时(秒)超过该值即视为病态视频")
    ap.add_argument("--logs", nargs="*", default=None,
                    help="指定日志文件; 默认扫描 output/*.log")
    args = ap.parse_args()

    logs = args.logs or sorted(glob.glob(os.path.join(OUT, "*.log")))
    mx = collections.defaultdict(float)     # basename -> max time
    cnt = collections.defaultdict(int)      # basename -> slow(>10s) count
    kind = {}                               # basename -> train_videos / teacher_cache

    for lg in logs:
        try:
            f = open(lg, errors="ignore")
        except OSError:
            continue
        with f:
            for line in f:
                m = PAT.search(line)
                if not m:
                    continue
                path, t = m.group(1), float(m.group(2))
                bn = path.split("/")[-1]
                kind[bn] = "teacher_cache" if "/teacher_cache/" in path else "train_videos"
                if t > mx[bn]:
                    mx[bn] = t
                if t > 10:
                    cnt[bn] += 1

    ranked = sorted(mx.items(), key=lambda x: -x[1])

    # 完整报告
    report_path = os.path.join(OUT, "slow_videos_report.txt")
    with open(report_path, "w") as w:
        w.write(f"# 扫描日志: {len(logs)} 个; 视频总数: {len(mx)}\n")
        w.write(f"# 阈值: {args.threshold}s\n")
        w.write("# basename\tmax_decode_s\tslow_cnt\tkind\n")
        for bn, v in ranked:
            w.write(f"{bn}\t{v:.1f}\t{cnt[bn]}\t{kind[bn]}\n")

    # 需转码的源视频 (train_videos 且 max > threshold)
    slow_src = [bn for bn, v in ranked
                if v > args.threshold and kind[bn] == "train_videos"]
    list_path = os.path.join(OUT, "slow_source_videos.txt")
    with open(list_path, "w") as w:
        for bn in slow_src:
            w.write(bn + "\n")

    slow_tc = [bn for bn, v in ranked
               if v > args.threshold and kind[bn] == "teacher_cache"]

    print(f"扫描日志 {len(logs)} 个, 视频总数 {len(mx)}")
    print(f">{args.threshold:.0f}s 源视频(需转码): {len(slow_src)}  -> {list_path}")
    print(f">{args.threshold:.0f}s teacher_cache: {len(slow_tc)} (转码源后清空缓存即可重建)")
    print(f"完整报告: {report_path}")


if __name__ == "__main__":
    main()
