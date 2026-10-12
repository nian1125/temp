#!/usr/bin/env bash
# ============================================================================
# 【此文件不属于 v3 快照 —— 已被守卫 stub 覆盖】
#
# run_opd_tvg_h20_fixed_all_152_v4.sh 是主线 (Temporal-ZwZ) 的脚本, 它依赖
# v4 才有的代码:
#   - OPDGRPOTrainer._repeat_visual_kwargs   (修复 teacher rollout 崩溃)
#   - _create_teacher_video(zoom_jitter=...) (非可逆粗提示)
#   - run_opd_tvg.sh 的 --zoom_jitter / --max_grad_norm 开关
# 这些在 v3 快照里【故意】不存在 (v3 = 冻结的历史状态)。
#
# 请到主线跑 v4:
#   bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/run_opd_tvg_h20_fixed_all_152_v4.sh
# ============================================================================
echo "[ERROR] 此脚本不属于 v3 快照。" >&2
echo "        v3 快照只用于复现 run_v3_s225 的历史结果, 不含 v4 的代码修复。" >&2
echo "        请到主线执行 v4:" >&2
echo "          bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ/scripts/run_opd_tvg_h20_fixed_all_152_v4.sh" >&2
exit 1
