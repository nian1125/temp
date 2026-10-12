#!/usr/bin/env python3
"""【此文件不属于 v3 快照 —— 已被守卫 stub 覆盖】

probe_teacher_channel.py 依赖 v4 才有的
    _create_teacher_video(..., zoom_jitter=...)
参数, 在 v3 快照的 tvg_opd_dataset.py 里不存在 (会 TypeError)。

请到主线运行:
    python3 /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/probe_teacher_channel.py ...
"""
import sys

print("[ERROR] 此脚本不属于 v3 快照。", file=sys.stderr)
print("        v3 快照是冻结的历史代码, 不含 zoom_jitter 等 v4 新增能力。", file=sys.stderr)
print("        请到主线执行:", file=sys.stderr)
print("          python3 /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/"
      "probe_teacher_channel.py --help", file=sys.stderr)
sys.exit(1)
