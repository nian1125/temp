#!/bin/bash
# 把 primus 上所有已聚合的 TVG 评测预测 jsonl 传到 OSS, 供本机挑 bad case。
set -e
OSS=oss://quark-llm/primus/datasets/qianzi.qz/czy_transfer/badcases/preds
TL=/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens
OPD=/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo
echo "=== 搜索已有预测文件 (charades/activitynet-timelens.jsonl) ==="
mapfile -t FILES < <(find "$TL/logs" "$OPD" /primus_xpfs_root/zhangyajie/czy -maxdepth 6 \
    -name "*-timelens.jsonl" 2>/dev/null | sort)
if [ ${#FILES[@]} -eq 0 ]; then echo "未找到任何 *-timelens.jsonl 预测文件"; exit 0; fi
for f in "${FILES[@]}"; do
    # 用 父目录名(含model_tag+时间) + 文件名 作为 OSS 上的唯一名
    tag=$(basename "$(dirname "$f")")
    name="${tag}__$(basename "$f")"
    sz=$(du -h "$f" | cut -f1)
    echo "[UP $sz] $f  ->  $name"
    ossutil cp -f "$f" "$OSS/$name" >/dev/null 2>&1 && echo "   ✅" || echo "   ❌"
done
echo "=== 完成, 共 ${#FILES[@]} 个文件 -> $OSS ==="
