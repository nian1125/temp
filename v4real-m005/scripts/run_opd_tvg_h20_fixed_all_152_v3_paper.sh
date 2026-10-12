#!/usr/bin/env bash
# ============================================================================
# 【此文件已被守卫 stub 覆盖】
#
# 这是从 v3 快照继承过来的 v3 启动脚本。在 Temporal-ZwZ-v4 目录里跑它会造成混淆:
# 脚本名说 v3, 但加载的是 v4 代码 (含 controller 修复) → 归因错乱。
#
# 请按需选择:
#   复现 v3 (controller 崩溃/不生效的历史状态):
#     bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v3/scripts/run_opd_tvg_h20_fixed_all_152_v3_paper.sh
#
#   跑 v4 (v3 + controller 修复):
#     bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4/scripts/run_opd_tvg_h20_v4_controller.sh
# ============================================================================
echo "[ERROR] 不要在 v4 目录里跑 v3 脚本 (名字与实际代码不符, 会导致归因错乱)。" >&2
echo "        复现 v3:  bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v3/scripts/run_opd_tvg_h20_fixed_all_152_v3_paper.sh" >&2
echo "        跑   v4:  bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4/scripts/run_opd_tvg_h20_v4_controller.sh" >&2
exit 1
