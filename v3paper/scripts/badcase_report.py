#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bad-case诊断报告(纯CPU, 单个自包含HTML, 内嵌可播放视频)。
读 canonical TimeLens eval 的 charades-timelens.jsonl, 挑 tIoU 最差的 N 个:
  - 内嵌【对应视频】(压成360p无音轨, 可播放可拖)
  - 交互时间轴: GT(绿)/预测(红)标在进度条上, 按钮一键跳到GT/预测起点, 播放头实时
  - 抽帧条(GT帧绿框/预测帧红框)快速一览
  - 完整模型输出文本 + 自动归因标签
目的: 看模型【错在哪、为什么错】(不是改benchmark)。

用法(primus登录节点, 纯CPU):
  python3 badcase_report.py --pred <charades-timelens.jsonl> --label TimeLens-8B \
    --out_html output/badcase.html --topk 40 --frames 10
带视频默认开; 关视频(只抽帧, 更小): --no_video
近失桶: --min_iou 0.01 --max_iou 0.5
"""
import argparse, ast, base64, io, json, os, re, subprocess, tempfile
import numpy as np


def load_preds(path):
    txt = open(path).read().strip()
    items = []
    try:
        obj = json.loads(txt)
        items = obj if isinstance(obj, list) else [obj]
    except Exception:
        for line in txt.splitlines():
            line = line.strip().rstrip(",")
            if line in ("", "[", "]"):
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    out = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        for k, v in it.items():
            parts = k.split(">>>")
            if len(parts) < 3:
                continue
            name, query, gtstr = parts[0], parts[1], ">>>".join(parts[2:])
            try:
                gt = ast.literal_eval(gtstr)
            except Exception:
                gt = None
            if gt and isinstance(gt[0], (list, tuple)):
                gt = list(gt[0])
            ts = v.get("timestamps") or []
            pred = list(ts[0]) if ts else None
            out[(name, query)] = {
                "gt": [float(gt[0]), float(gt[1])] if gt else None,
                "pred": [float(pred[0]), float(pred[1])] if pred else None,
                "ans": v.get("answers", ""),
                "dur": float(v.get("duration", 0) or 0),
            }
    return out


def iou(a, b):
    if not a or not b:
        return 0.0
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    u = max(a[1], b[1]) - min(a[0], b[0])
    return max(hi - lo, 0.0) / u if u > 0 else 0.0


def overlaps(a, b):
    return a and b and min(a[1], b[1]) > max(a[0], b[0])


def tag_case(d, dur, iou1):
    """归因标签: 帮你快速判断错误类型。"""
    tags = []
    gt, pred = d["gt"], d["pred"]
    glen = (gt[1] - gt[0]) if gt else 0
    if pred is None or (dur and pred[0] >= dur + 5):
        tags.append(("no-pred", "模型没给有效时间(格式/拒答)"))
    else:
        if iou1 < 1e-4:
            if overlaps(gt, pred):
                tags.append(("touch-miss", "沾边但几乎不重叠"))
            else:
                tags.append(("wrong-region", "定到完全不同区域=找错地方"))
        elif iou1 < 0.5:
            tags.append(("near-miss", "部分重叠、边界不准=精度问题"))
        # 偏宽/偏窄/偏早/偏晚(系统性偏差诊断)
        if pred and gt:
            plen = pred[1] - pred[0]
            if glen > 0 and plen > glen * 1.6:
                tags.append(("pred-too-wide", "预测区间比GT宽很多"))
            elif glen > 0 and plen < glen * 0.6:
                tags.append(("pred-too-narrow", "预测区间比GT窄很多"))
            mid_p, mid_g = (pred[0]+pred[1])/2, (gt[0]+gt[1])/2
            if mid_p < mid_g - max(glen*0.5, 2):
                tags.append(("pred-early", "预测整体偏早"))
            elif mid_p > mid_g + max(glen*0.5, 2):
                tags.append(("pred-late", "预测整体偏晚"))
    if gt and dur:
        if glen / dur > 0.85:
            tags.append(("gt~whole-video", "GT≈整段视频"))
        if glen < 1.0:
            tags.append(("gt-tiny", "GT<1秒(帧率下难分辨)"))
    return tags


def frame_strip(vr, dur, gt, pred, n_frames, width=140, pred2=None):
    from PIL import Image, ImageDraw
    total = len(vr)
    fps = vr.get_avg_fps() or (total / dur if dur else 2.0)
    times = np.linspace(0, max(dur - 1e-3, 0), n_frames)
    idxs = [min(int(t * fps), total - 1) for t in times]
    try:
        batch = vr.get_batch(idxs).asnumpy()
    except Exception:
        batch = np.stack([vr[i].asnumpy() for i in idxs])
    res = []
    for t, arr in zip(times, batch):
        im = Image.fromarray(arr).convert("RGB")
        w, h = im.size
        im = im.resize((width, int(h * width / w)))
        cols = []
        if gt and gt[0] <= t <= gt[1]:
            cols.append((63, 185, 80))      # 绿=GT
        if pred and pred[0] <= t <= pred[1]:
            cols.append((248, 81, 73))      # 红=pred1
        if pred2 and pred2[0] <= t <= pred2[1]:
            cols.append((78, 161, 255))     # 蓝=pred2
        if cols:
            dr = ImageDraw.Draw(im)
            for bi, c in enumerate(cols):
                dr.rectangle([bi*2, bi*2, im.size[0]-1-bi*2, im.size[1]-1-bi*2], outline=c, width=3)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=68)
        res.append((float(t), base64.b64encode(buf.getvalue()).decode()))
    return res


_FFMPEG = None
def ffmpeg_exe():
    global _FFMPEG
    if _FFMPEG is not None:
        return _FFMPEG
    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        _FFMPEG = "ffmpeg"
    return _FFMPEG


def video_b64(vpath, vmax_w=360):
    """转码成360p无音轨h264, 返回base64; 失败返回None。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cmd = [ffmpeg_exe(), "-y", "-i", vpath,
           "-vf", f"scale={vmax_w}:-2", "-an",
           "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
           "-movflags", "+faststart", tmp.name]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120, check=True)
        data = open(tmp.name, "rb").read()
        if len(data) < 100:
            raise RuntimeError("empty")
        return base64.b64encode(data).decode()
    except Exception:
        # 兜底: 直接塞原文件(可能较大)
        try:
            return base64.b64encode(open(vpath, "rb").read()).decode()
        except Exception:
            return None
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--pred2", default=None, help="第二个模型预测(蓝色叠加, 如你的OPSD)")
    ap.add_argument("--label2", default="model2")
    ap.add_argument("--video_root",
                    default="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens/data/TimeLens-Bench/videos/charades")
    ap.add_argument("--out_html", required=True)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--min_iou", type=float, default=-1.0)
    ap.add_argument("--max_iou", type=float, default=1.01)
    ap.add_argument("--no_video", action="store_true", help="不内嵌视频(只抽帧, HTML更小)")
    ap.add_argument("--vmax_w", type=int, default=360)
    args = ap.parse_args()

    import decord
    decord.bridge.set_bridge("native")
    from decord import VideoReader, cpu

    P1 = load_preds(args.pred)
    P2 = load_preds(args.pred2) if args.pred2 else {}
    print(f"{args.label}: {len(P1)} 条" + (f" | {args.label2}: {len(P2)} 条" if P2 else ""))
    rows = []
    for key, d in P1.items():
        i1 = iou(d["gt"], d["pred"])
        if not (args.min_iou <= i1 <= args.max_iou):
            continue
        d2 = P2.get(key)
        i2 = iou(d2["gt"], d2["pred"]) if d2 else None
        rows.append((key, d, i1, d2, i2))
    rows.sort(key=lambda r: r[2])
    rows = rows[:args.topk]
    print(f"取最差 {len(rows)} 个 (iou∈[{args.min_iou},{args.max_iou}], 按{args.label}排序)"
          + ("" if args.no_video else " · 内嵌视频(转码中, 稍慢)"))

    cards, tagcount = [], {}
    for ci, (key, d, i1, d2, i2) in enumerate(rows):
        name, query = key
        dur = d["dur"]
        vpath = os.path.join(args.video_root, name if name.endswith(".mp4") else name + ".mp4")
        tags = tag_case(d, dur, i1)
        # 两模型都偏GT却彼此一致 = 共同难点(不是单模型菜)
        if d2 and i1 < 0.3 and i2 < 0.3 and overlaps(d["pred"], d2["pred"]) and iou(d["pred"], d2["pred"]) > 0.4:
            tags.append(("both-agree-vs-GT", "两模型一致但都偏GT=共同难点/数据歧义"))
        for t, _ in tags:
            tagcount[t] = tagcount.get(t, 0) + 1
        pred2span = d2["pred"] if d2 else None
        strip_html, vid_html = "", ""
        try:
            vr = VideoReader(vpath, ctx=cpu(0))
            if not dur:
                dur = len(vr) / (vr.get_avg_fps() or 2.0)
            for t, b64 in frame_strip(vr, dur, d["gt"], d["pred"], args.frames, pred2=pred2span):
                strip_html += (f'<div class="fr"><img src="data:image/jpeg;base64,{b64}">'
                               f'<div class="ft">{t:.1f}s</div></div>')
        except Exception as e:
            strip_html = f'<div class="err">⚠ 视频/帧读取失败: {vpath} ({str(e)[:80]})</div>'
        if not args.no_video:
            vb = video_b64(vpath, args.vmax_w)
            if vb:
                g0 = d["gt"][0] if d["gt"] else 0; g1 = d["gt"][1] if d["gt"] else 0
                p0 = d["pred"][0] if d["pred"] else 0; p1 = d["pred"][1] if d["pred"] else 0
                q0 = pred2span[0] if pred2span else 0; q1 = pred2span[1] if pred2span else 0
                seg2 = '<div class="seg p2"></div>' if d2 else ''
                btn2 = (f'<button onclick="seek(this,\'q0\')">▶ {args.label2}起点 {q0:.1f}s</button>'
                        f'<button onclick="seek(this,\'q1\')">{args.label2}终点 {q1:.1f}s</button>') if d2 else ''
                vid_html = f"""
  <div class="player">
    <video class="vid" preload="metadata" controls
           data-g0="{g0}" data-g1="{g1}" data-p0="{p0}" data-p1="{p1}" data-q0="{q0}" data-q1="{q1}" data-has2="{1 if d2 else 0}"
           src="data:video/mp4;base64,{vb}"></video>
    <div class="tl"><div class="seg gt"></div><div class="seg p1"></div>{seg2}<div class="ph"></div></div>
    <div class="seekbtns">
      <button onclick="seek(this,'g0')">▶ GT起点 {g0:.1f}s</button>
      <button onclick="seek(this,'g1')">GT终点 {g1:.1f}s</button>
      <button onclick="seek(this,'p0')">▶ {args.label}起点 {p0:.1f}s</button>
      <button onclick="seek(this,'p1')">{args.label}终点 {p1:.1f}s</button>
      {btn2}
    </div>
  </div>"""
        tagspan = "".join(f'<span class="tag t-{t.split("~")[0].split("-")[0]}">{t}</span>' for t, _ in tags)
        tagtip = " · ".join(f"{t}: {desc}" for t, desc in tags)
        dtags = " ".join(t for t, _ in tags)
        p2_span = (f'<span class="m2">{args.label2}: {d2["pred"]} IoU=<b>{i2:.3f}</b></span>') if d2 else ''
        p2_ans = (f'<details><summary>完整模型输出({args.label2})</summary>'
                  f'<pre>{(d2["ans"] or "").replace("<","&lt;")}</pre></details>') if d2 else ''
        cards.append(f"""
<div class="card" data-tags="{dtags}">
  <div class="hd"><span class="idx">#{ci+1}</span>
    <span class="iou {'bad' if i1<0.3 else 'mid' if i1<0.5 else 'ok'}">{args.label} IoU {i1:.3f}</span>
    {f'<span class="iou {chr(98) if i2<0.3 else chr(109)}2">{args.label2} {i2:.3f}</span>' if d2 else ''}
    {tagspan}<span class="vid2">{name} · dur {dur:.1f}s</span></div>
  <div class="q">“{query}”</div>
  <div class="spans"><span class="gt">GT: {d['gt']}</span>
    <span class="m1">{args.label}: {d['pred']}</span> {p2_span}</div>
  <div class="tagtip">{tagtip}</div>
  {vid_html}
  <details><summary>抽帧一览(绿框=GT / 红框={args.label}{' / 蓝框='+args.label2 if d2 else ''})</summary><div class="strip">{strip_html}</div></details>
  <details><summary>完整模型输出({args.label})</summary><pre>{(d['ans'] or '').replace('<','&lt;')}</pre></details>
  {p2_ans}
</div>""")

    summ = " ".join(f'<span class="tag">{t}: {c}</span>' for t, c in sorted(tagcount.items(), key=lambda x:-x[1]))
    btns = '<button onclick="flt(\'\')">全部</button>' + "".join(
        f'<button onclick="flt(\'{t}\')">{t} ({c})</button>' for t, c in sorted(tagcount.items(), key=lambda x:-x[1]))
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<title>Bad-case诊断 · {args.label}</title><style>
body{{margin:0;background:#0f1419;color:#e6edf3;font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;font-size:14px}}
header{{position:sticky;top:0;background:#0c1116;border-bottom:1px solid #2d3a48;padding:12px 20px;z-index:9}}
h1{{font-size:18px;margin:0 0 6px}} .sum{{color:#9aa7b4;font-size:13px}}
button{{background:#222b38;color:#e6edf3;border:1px solid #2d3a48;border-radius:6px;padding:4px 10px;margin:2px;cursor:pointer;font-size:12px}}
button:hover{{background:#2d3a48}}
main{{padding:16px 20px;max-width:1100px;margin:0 auto}}
.card{{background:#1a212b;border:1px solid #2d3a48;border-radius:12px;padding:12px 16px;margin:14px 0}}
.hd{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}} .idx{{color:#9aa7b4;font-weight:700}}
.vid2{{color:#9aa7b4;font-size:12px;margin-left:auto}}
.iou{{font-weight:700;padding:1px 8px;border-radius:20px}} .iou.bad{{background:rgba(248,81,73,.18);color:#f85149}}
.iou.mid{{background:rgba(210,153,34,.18);color:#d29922}} .iou.ok{{background:rgba(63,185,80,.18);color:#3fb950}}
.iou.b2{{background:rgba(78,161,255,.18);color:#4ea1ff}} .iou.m2{{background:rgba(78,161,255,.12);color:#7fb8ff}}
.tag{{font-size:11.5px;padding:1px 8px;border-radius:20px;background:#222b38;color:#9ecbff}}
.t-gt{{background:rgba(210,153,34,.2);color:#e3b341}} .t-pred{{background:rgba(163,113,247,.2);color:#c9a9ff}}
.t-both{{background:rgba(248,81,73,.22);color:#ff9b94;font-weight:700}}
.q{{font-size:15px;margin:8px 0;color:#fff}}
.spans{{font-size:12.5px;display:flex;gap:16px;flex-wrap:wrap}} .gt{{color:#3fb950}} .m1{{color:#f85149}} .m2{{color:#4ea1ff}}
.tagtip{{color:#6e7b8a;font-size:11.5px;margin:4px 0}}
.player{{margin:8px 0}} .vid{{width:100%;max-width:640px;border-radius:8px;background:#000;display:block}}
.tl{{position:relative;height:18px;max-width:640px;background:#0c1116;border-radius:7px;margin:4px 0;overflow:hidden}}
.seg{{position:absolute;height:33%;opacity:.7}} .seg.gt{{top:0;background:#3fb950}} .seg.p1{{top:34%;background:#f85149}} .seg.p2{{top:67%;background:#4ea1ff}}
.ph{{position:absolute;top:0;width:2px;height:100%;background:#fff;left:0}}
.seekbtns button{{font-size:11.5px}}
.strip{{display:flex;gap:4px;overflow-x:auto;padding:8px 0}} .fr{{flex:0 0 auto;text-align:center}}
.fr img{{border-radius:4px;display:block}} .ft{{font-size:10px;color:#9aa7b4}}
details{{margin-top:6px}} summary{{cursor:pointer;color:#9ecbff;font-size:12.5px}}
pre{{background:#0c1116;border:1px solid #2d3a48;border-radius:6px;padding:8px;white-space:pre-wrap;font-size:12px;max-height:240px;overflow:auto}}
.err{{color:#f85149;font-size:12px}}
</style></head><body>
<header><h1>Bad-case诊断 · {args.label}{(' vs '+args.label2) if args.pred2 else ''} · Charades-TimeLens</h1>
<div class="sum">最差 {len(rows)} 个(按{args.label} IoU升序) · <span style="color:#3fb950">■GT</span> <span style="color:#f85149">■{args.label}</span>{(' <span style=color:#4ea1ff>■'+args.label2+'</span>') if args.pred2 else ''} · 标签: {summ or '无'}</div>
<div style="margin-top:6px">{btns}</div></header>
<main>{''.join(cards)}</main>
<script>
function seek(b,k){{const v=b.closest('.player').querySelector('video');v.currentTime=+v.dataset[k]||0;v.play();}}
document.querySelectorAll('video.vid').forEach(v=>{{
  function lay(){{const D=v.duration||1,pl=v.parentElement;
    const g=pl.querySelector('.seg.gt'),p=pl.querySelector('.seg.p1'),q=pl.querySelector('.seg.p2');
    g.style.left=(v.dataset.g0/D*100)+'%';g.style.width=Math.max((v.dataset.g1-v.dataset.g0)/D*100,0.5)+'%';
    p.style.left=(v.dataset.p0/D*100)+'%';p.style.width=Math.max((v.dataset.p1-v.dataset.p0)/D*100,0.5)+'%';
    if(q&&+v.dataset.has2){{q.style.left=(v.dataset.q0/D*100)+'%';q.style.width=Math.max((v.dataset.q1-v.dataset.q0)/D*100,0.5)+'%';}}}}
  v.addEventListener('loadedmetadata',lay);
  v.addEventListener('timeupdate',()=>{{const D=v.duration||1;
    v.parentElement.querySelector('.ph').style.left=(v.currentTime/D*100)+'%';}});
}});
function flt(t){{document.querySelectorAll('.card').forEach(c=>{{
  c.style.display=(!t||c.dataset.tags.split(' ').includes(t))?'':'none';}});}}
</script></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(args.out_html)), exist_ok=True)
    open(args.out_html, "w").write(html)
    print(f"\n✅ 报告: {args.out_html}  ({len(html)//1024//1024 or 1} MB级)")
    print(f"标签统计: {tagcount}")


if __name__ == "__main__":
    main()
