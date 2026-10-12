#!/bin/bash
# 【primus登录节点】跑(纯CPU)。生成内嵌可播放视频的bad-case诊断HTML, 传回OSS。
# 默认: TimeLens-8B。可叠加第二模型(你的OPSD): export PRED2=<jsonl> LABEL2=MyOPSD
# 其他: export PRED=<jsonl> LABEL=xx TOPK=35 MINIOU=-1 MAXIOU=1.01 NAME=badcase
set -e
ROOT=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
cd $ROOT
pip install decord pillow numpy imageio-ffmpeg 2>/dev/null
PRED=${PRED:-$(ls -t $ROOT/output/eval-timelens8b-official/*/charades-timelens.jsonl 2>/dev/null | head -1)}
[ -z "$PRED" ] && { echo "❌ 没找到预测文件, 先设 PRED 或跑 submit_eval_timelens8b.sh"; exit 1; }
NAME=${NAME:-badcase}
ARGS="--pred $PRED --label ${LABEL:-TimeLens-8B} --topk ${TOPK:-35} --frames 10 --min_iou ${MINIOU:--1} --max_iou ${MAXIOU:-1.01} --out_html $ROOT/output/${NAME}.html"
if [ -n "$PRED2" ]; then ARGS="$ARGS --pred2 $PRED2 --label2 ${LABEL2:-MyOPSD}"; echo "叠加第二模型: ${LABEL2:-MyOPSD} = $PRED2"; fi
echo "PRED=$PRED  TOPK=${TOPK:-35}  IoU∈[${MINIOU:--1},${MAXIOU:-1.01}]  NAME=$NAME"
python3 scripts/badcase_report.py $ARGS
OSS=oss://quark-llm/primus/datasets/qianzi.qz/czy_transfer/badcases
C=/primus_xpfs_root/zhangyajie/czy/a_uploadoss/ossutilconfig.txt
ossutil cp -f $ROOT/output/${NAME}.html $OSS/${NAME}.html -c $C
echo "✅ 传 $OSS/${NAME}.html"
