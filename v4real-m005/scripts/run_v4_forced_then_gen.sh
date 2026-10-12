#!/usr/bin/env bash
# 串行: v4-forced (新v1, teacher用学生前缀) -> v4-generate (teacher独立采样)
# 两个实验同框架(controller=paper), 唯一变量=teacher能力估计方式, 验证Claude发现的forced乐观偏差
# 实测(forced run 84步): λ=0占62.8%/λ=1占37.2%, controller正常按设计工作, 蒸馏有效
# 用法: nohup bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4/scripts/run_v4_forced_then_gen.sh >> /mnt/250T_ceph/zhenjunchen/logs/v4_forced_then_gen.log 2>&1 &
set -euo pipefail

V4=/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4

echo "===== [1/2] v4-forced (新v1: teacher_tiou_mode=forced) ====="
bash "${V4}/scripts/run_opd_tvg_h20_v4_controller.sh"

echo "===== [2/2] v4-generate (teacher独立采样) ====="
bash "${V4}/scripts/run_opd_tvg_h20_v4_controller_gen.sh"

echo "===== 全部完成: $(date) ====="
