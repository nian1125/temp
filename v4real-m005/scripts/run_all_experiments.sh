#!/usr/bin/env bash
# 串行跑 3 个实验(共用8卡, 一个跑完再跑下一个). 配合 nohup 后台运行.
# 用法(后台):
#     cd /mnt/250T_ceph/zhenjunchen/code/Temporal-ZwZ
#     nohup bash scripts/run_all_experiments.sh > output/all_experiments.log 2>&1 &
#     echo $! > output/all_exp.pid      # 记录主进程PID
# 查看进度: tail -f output/all_experiments.log
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

run_one() {
    local name="$1" script="$2"
    echo ""
    echo "########################################################"
    echo "# 开始 ${name}  $(date '+%F %T')"
    echo "########################################################"
    bash "${SCRIPT_DIR}/${script}" 2>&1 | tee "output/${name}.log"
    echo "# 完成 ${name}  $(date '+%F %T'), 退出码 ${PIPESTATUS[0]}"
    # 每个实验之间清空 teacher_cache, 避免跨实验污染
    rm -f /dockerdata/zhenjunchen/teacher_cache/*.mp4 2>/dev/null || true
    sleep 30   # 等 GPU 显存释放干净
}

# 实验2: coarse_box 1.25 (蒸馏)
run_one "exp2_zoom1.25"  "run_opd_tvg_h20_exp2.sh"
# 实验3: coarse_box 1.75 (蒸馏)
run_one "exp3_zoom1.75"  "run_opd_tvg_h20_exp3.sh"
# 实验4: 纯GRPO kl=0 (冻结teacher, 只训student)
run_one "exp4_pure_grpo" "run_opd_tvg_h20_exp4.sh"

echo ""
echo "======== 全部 3 个实验完成 $(date '+%F %T') ========"
