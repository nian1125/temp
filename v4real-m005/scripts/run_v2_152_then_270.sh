#!/usr/bin/env bash
# 串行: 等当前 v2-152 (max_steps=225) 跑完 -> 自动启动 v2-270
# 用法: nohup bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/run_v2_152_then_270.sh >> /mnt/250T_ceph/zhenjunchen/logs/v2_152_then_270.log 2>&1 &
set -euo pipefail

SCRIPTS=/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts

echo "===== [1/2] 等待 v2-152 (fixed-all-152-v2-s225) 跑完 ====="
# 轮询: 进程消失且日志出现完成标志
while true; do
    if ! pgrep -f "train_opd_tvg.py" >/dev/null 2>&1; then
        # 进程没了, 再确认日志有完成标志(防误杀)
        if grep -aq "fixed_all+152 v2 训练完成\|train_runtime\|Done\. Output" /mnt/250T_ceph/zhenjunchen/logs/run_v2_s225.log 2>/dev/null; then
            echo "v2-152 已正常完成 ✅"
            break
        else
            echo "v2-152 进程消失但无完成标志, 可能被kill或崩溃, 不启动270" >&2
            exit 1
        fi
    fi
    sleep 60
done

echo "===== [2/2] 启动 v2-270 (fixed-all-270-v2-s225) ====="
bash "${SCRIPTS}/run_opd_tvg_h20_fixed_all_270_v2.sh"

echo "===== 全部完成: $(date) ====="
