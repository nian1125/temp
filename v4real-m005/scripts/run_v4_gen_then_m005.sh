#!/usr/bin/env bash
# ============================================================================
# 串行: [1/2] v4-generate (teacher 独立采样)  ->  [2/2] v4-m005 (forced + 放宽margin)
#
# 两个实验同框架 (controller=paper, 152数据, 225步), 唯一变量 = teacher 能力估计/判定:
#   [1/2] gen : --teacher_tiou_mode generate      (teacher 独立 rollout, λ判定最真实)
#   [2/2] m005: --teacher_tiou_mode forced        (teacher-forced, student前缀锚定)
#               --teacher_advantage_margin -0.05  (放宽"有优势"门槛)
#
# 背景: 首轮 forced + margin=0.0 实测 λ=0 占 62.8% (蒸馏只在37%组生效), 评测效果不好。
#   margin=-0.05 后判定变为 no_edge = r̄T <= r̄S - 0.05, 即 teacher 需明显更差才不蒸馏,
#   λ=0 比例应显著下降 → 蒸馏组变多。
#   注: v4 代码里 teacher_tiou_mode 只有 generate 一条独立分支, argmax/forced 同走
#   _teacher_forced_group_tiou, 故不存在"改回argmax"的开关, 只能靠 margin 调节。
#
# ----------------------------------------------------------------------------
# ★ 全部逻辑包在 main() 里, 最后一行才调用。
#   原因: bash 是流式读脚本的。上一版串行脚本在 [1/2] 训练阻塞的9小时里被编辑过,
#   文件字节偏移错乱 → 第一步跑完后 bash 从旧偏移继续读, 把注释残片当命令执行:
#       run_v4_forced_then_gen.sh: line 11: syntax error near unexpected token `新v1:'
#   → [2/2] 从未启动, 8卡白闲一整夜。
#   包成函数后 bash 会先完整解析函数体, 运行期编辑不再影响后续执行。
#   即便如此: 【本脚本运行期间请勿编辑】。
# ----------------------------------------------------------------------------
# 用法:
#   nohup bash /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4/scripts/run_v4_gen_then_m005.sh \
#       >> /mnt/250T_ceph/zhenjunchen/logs/v4_gen_then_m005.log 2>&1 &
#
# 预期: gen ~8-10h (teacher独立采样慢, 且有 hang 风险), m005 ~9h。
#   gen 若 hang 被 watchdog 杀, 脚本会继续跑 m005 (不 set -e 中断第二步)。
# ============================================================================
set -uo pipefail

main() {
    local V4=/mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ-v4
    local rc_gen=0 rc_m005=0

    echo "############################################################"
    echo "# 串行任务启动: $(date)"
    echo "# [1/2] v4-generate   (teacher独立采样)"
    echo "# [2/2] v4-m005       (forced + margin=-0.05)"
    echo "############################################################"

    echo ""
    echo "===== [1/2] v4-generate 开始: $(date) ====="
    bash "${V4}/scripts/run_opd_tvg_h20_v4_controller_gen.sh" || rc_gen=$?
    if [ ${rc_gen} -ne 0 ]; then
        echo "!!!!! [1/2] v4-generate 非零退出 (rc=${rc_gen}) — 已知 gen 模式有 NCCL hang 风险"
        echo "!!!!! 继续执行 [2/2], 不因此中断整个串行"
    fi
    echo "===== [1/2] v4-generate 结束: $(date) (rc=${rc_gen}) ====="

    # 等 GPU 显存/NCCL 端口完全释放, 避免第二步的 pgrep 自检误判或抢显存
    echo ">>> 等待 180s 让 GPU 与 NCCL 资源释放 ..."
    sleep 180
    pkill -9 -f "train_opd_tvg.py" 2>/dev/null || true
    sleep 20

    echo ""
    echo "===== [2/2] v4-m005 (margin=-0.05) 开始: $(date) ====="
    bash "${V4}/scripts/run_opd_tvg_h20_v4_controller.sh" || rc_m005=$?
    echo "===== [2/2] v4-m005 结束: $(date) (rc=${rc_m005}) ====="

    echo ""
    echo "############################################################"
    echo "# 全部完成: $(date)"
    echo "#   [1/2] v4-generate rc=${rc_gen}"
    echo "#   [2/2] v4-m005     rc=${rc_m005}"
    echo "############################################################"
}

main "$@"
