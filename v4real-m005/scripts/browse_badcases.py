#!/usr/bin/env python3
"""Bad-case 浏览器: 读 TimeLens 评测预测 jsonl → 算 tIoU → 最差排前 → 生成可看视频的 HTML。

预测文件格式(TimeLens eval_dataloader.py dump): list of
  {"{video_name}>>>{query}>>>{span}": {"timestamps": [[s,e],...], "answers": "...", "duration": d}}

用法:
  python browse_badcases.py \
    --pred /path/to/charades-timelens.jsonl \
    --video_dir /cpfs01/user/chenzhaoyu.czy/data/charades/raw_480p/Charades_v1_480 \
    --topn 50 --max_iou 0.3 --out /cpfs01/user/chenzhaoyu.czy/badcases.html
然后在 VSCode 里打开 badcases.html (或浏览器), 每条 bad case 直接内嵌播放视频。
"""
import argparse, ast, html, json, os, re


def parse_span(s):
    """span 字段是 python-repr, 可能是 [s,e] / [[s,e]] / [[s,e],...]。取第一个区间。"""
    try:
        v = ast.literal_eval(s) if isinstance(s, str) else s
    except Exception:
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", str(s))]
        return [nums[0], nums[1]] if len(nums) >= 2 else None
    while isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
        v = v[0]
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return [float(v[0]), float(v[1])]
    return None


def iou(a, b):
    if not a or not b:
        return 0.0
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    u = max(a[1], b[1]) - min(a[0], b[0])
    return max(hi - lo, 0.0) / u if u > 0 else 0.0


def load(pred_path):
    recs = []
    raw = json.load(open(pred_path)) if pred_path.endswith(".json") else \
        [json.loads(l) for l in open(pred_path) if l.strip()]
    # 文件可能是 list[dict单键] 也可能是 dict
    items = []
    if isinstance(raw, dict):
        items = list(raw.items())
    else:
        for d in raw:
            if isinstance(d, dict):
                items.extend(d.items())
    for key, val in items:
        parts = key.split(">>>")
        if len(parts) < 3:
            continue
        vid, query, span_s = parts[0], parts[1], ">>>".join(parts[2:])
        gt = parse_span(span_s)
        ts = val.get("timestamps") or []
        pred = None
        if ts:
            p = ts[0]
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pred = [float(p[0]), float(p[1])]
        recs.append({
            "video": vid, "query": query, "gt": gt, "pred": pred,
            "iou": round(iou(pred, gt), 4),
            "dur": val.get("duration"), "answer": val.get("answers", ""),
        })
    return recs


HTML_HEAD = """<!doctype html><meta charset=utf-8><title>Bad cases</title>
<style>
body{font-family:-apple-system,Segoe UI,Helvetica,sans-serif;margin:0;background:#111;color:#eee}
.hdr{position:sticky;top:0;background:#1c1c1c;padding:10px 16px;border-bottom:1px solid #333;z-index:9}
.card{display:flex;gap:16px;padding:16px;border-bottom:1px solid #222}
.card video{width:420px;border-radius:6px;background:#000}
.meta{flex:1;font-size:14px;line-height:1.7}
.q{font-size:16px;font-weight:600;margin-bottom:8px;color:#fff}
.bar{position:relative;height:22px;background:#222;border-radius:4px;margin:6px 0;overflow:hidden}
.gt{position:absolute;height:100%;background:#2e7d32aa;top:0}
.pr{position:absolute;height:100%;background:#c6282899;top:0;border-left:2px solid #ff5252;border-right:2px solid #ff5252}
.tag{display:inline-block;font-size:12px;padding:1px 6px;border-radius:3px;margin-right:6px}
.iou{font-weight:700}.lo{color:#ff6b6b}.mid{color:#ffd166}.hi{color:#06d6a0}
code{color:#9cdcfe}
</style>
"""


def bar(gt, pred, dur):
    if not dur or dur <= 0:
        dur = max([x for seg in [gt, pred] if seg for x in seg] + [1])
    def seg(s, cls):
        if not s:
            return ""
        l = max(0, s[0]) / dur * 100
        w = max(0.5, (s[1] - s[0]) / dur * 100)
        return f'<div class="{cls}" style="left:{l:.1f}%;width:{w:.1f}%"></div>'
    return f'<div class="bar">{seg(gt,"gt")}{seg(pred,"pr")}</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--video_dir", default="/cpfs01/user/chenzhaoyu.czy/data/charades/raw_480p/Charades_v1_480")
    ap.add_argument("--out", default="/cpfs01/user/chenzhaoyu.czy/badcases.html")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--max_iou", type=float, default=1.0, help="只看 iou<=该值 的样本")
    ap.add_argument("--min_dur", type=float, default=0.0)
    args = ap.parse_args()

    recs = load(args.pred)
    recs = [r for r in recs if r["iou"] <= args.max_iou and (r["dur"] or 0) >= args.min_dur]
    recs.sort(key=lambda r: r["iou"])
    sel = recs[: args.topn]
    miou = sum(r["iou"] for r in recs) / len(recs) if recs else 0.0

    rows = []
    n_missing = 0
    for i, r in enumerate(sel):
        vp = os.path.join(args.video_dir, r["video"])
        exists = os.path.exists(vp)
        if not exists:
            n_missing += 1
        cls = "lo" if r["iou"] < 0.3 else ("mid" if r["iou"] < 0.5 else "hi")
        vtag = (f'<video controls preload=none src="file://{html.escape(vp)}"></video>'
                if exists else f'<div style="width:420px;color:#888">视频缺失:<br><code>{html.escape(vp)}</code></div>')
        gt = r["gt"]; pr = r["pred"]
        rows.append(f"""<div class=card>{vtag}<div class=meta>
<div class=q>#{i+1} &nbsp;<span class="iou {cls}">tIoU={r['iou']:.3f}</span></div>
<div>{html.escape(r['query'])}</div>
<div><span class=tag style="background:#2e7d32">GT</span>{gt[0]:.1f} – {gt[1]:.1f}s
&nbsp;<span class=tag style="background:#c62828">Pred</span>{('%.1f – %.1f s'%(pr[0],pr[1])) if pr else '∅(未解析)'}
&nbsp; dur={r['dur']:.1f}s</div>
{bar(gt, pr, r['dur'])}
<div style="color:#888;font-size:12px">video: <code>{html.escape(r['video'])}</code></div>
</div></div>""")

    hdr = (f'<div class=hdr><b>Bad cases</b> — {os.path.basename(args.pred)} | '
           f'全集 {len(recs)} 条 (max_iou≤{args.max_iou}), mIoU={miou:.3f} | '
           f'展示最差 {len(sel)} 条 | 视频缺失 {n_missing}</div>')
    open(args.out, "w").write(HTML_HEAD + hdr + "".join(rows))
    print(f"✅ HTML: {args.out}")
    print(f"   全集 {len(recs)} 条, mIoU={miou:.3f}, 展示最差 {len(sel)} 条, 视频缺失 {n_missing}")
    print("\n最差 15 条:")
    for i, r in enumerate(sel[:15]):
        pr = ("%.1f-%.1f" % tuple(r["pred"])) if r["pred"] else "∅"
        gt = "%.1f-%.1f" % tuple(r["gt"]) if r["gt"] else "?"
        print(f"  {i+1:2d}. iou={r['iou']:.3f}  GT[{gt}] Pred[{pr}]  {r['video']}  | {r['query'][:60]}")


if __name__ == "__main__":
    main()
