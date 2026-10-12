#!/usr/bin/env bash
# 串行: 等 v3-270 (快照 Temporal-ZwZ-v3, 20260917-2100) 跑完 -> 自检v4 -> 启动 v4-152
# 用法: nohup bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4/scripts/run_after_v3_270.sh >> /mnt/250T_ceph/zhenjunchen/logs/v4_after_v3.log 2>&1 &
set -euo pipefail

V4=/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4
V3_LOG=/mnt/250T_ceph/zhenjunchen/logs/run_v3_270.log

echo "===== [1/3] 等待 v3-270 (快照版) 跑完 ====="
while true; do
    if ! pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; then
        if grep -aq "训练完成\|train_runtime\|Done\. Output" "$V3_LOG" 2>/dev/null; then
            echo "v3-270 已正常完成 ✅"
            break
        else
            echo "v3-270 进程消失但无完成标志, 可能被kill/崩溃, 不启动v4" >&2
            exit 1
        fi
    fi
    sleep 120
done

echo "===== [2/3] v4 自检 ====="
source /dockerdata/zhenjunchen/venv/bin/activate
cd "${V4}"
python3 scripts/verify_v4.py

echo "===== [3/3] 启动 v4-152 (fixed_all + 152数据, max_steps=225) ====="
bash "${V4}/scripts/run_opd_tvg_h20_v4_controller.sh"

echo "===== 全部完成: $(date) ====="
