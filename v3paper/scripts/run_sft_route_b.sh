#!/usr/bin/env bash
# ============================================================
# 路线β 第1步: SFT 热启动 (Qwen3-VL-8B)
# 在过滤数据上做轻量 SFT, 学会任务格式+基础定位, 给后续 OPD 一个热启动。
#
# ⚠️ 重要提醒(必读):
# 1. train_sft.py 原本是给 Qwen2.5-VL-7B 写的, 用 Qwen3-VL-8B 前请先小样本冒烟测试!
# 2. SFT 的视频token化(num_frames/max_seq_len)和 OPD(fps=2/total_tokens=14336)不同,
#    理想情况应对齐, 否则 SFT→OPD 有 train/train 不一致。本脚本已尽量调大 max_seq_len。
# 3. 建议先用前50条冒烟(--train_file 改成小文件), 确认能跑通再全量。
# ============================================================
set -euo pipefail

PROJECT_ROOT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3"
TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"

MODEL_PATH="${MODEL_PATH:-/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct}"
# SFT 数据: 默认用过滤的6838(干净、格式对、video_path相对TimeLens根可解析)
# 可换成更大的集合(如30K采样)做更强SFT
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl}"
OUTPUT_DIR="${PROJECT_ROOT}/output/sft/SFT-Qwen3-VL-8B-routeb"

# 视频路径在 train_file 里是相对 "data/TimeLens-100K/..." → cwd 设为 TimeLens 根才能解析
cd "$TIMELENS_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 减少显存碎片

echo "=========================================="
echo "SFT 路线β | model=$MODEL_PATH"
echo "  train_file=$TRAIN_FILE ($(wc -l < "$TRAIN_FILE") samples)"
echo "  output=$OUTPUT_DIR  (cwd=$TIMELENS_ROOT for video resolution)"
echo "=========================================="

torchrun --nproc_per_node=8 \
    "${PROJECT_ROOT}/scripts/train_sft.py" \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 2 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --max_seq_len 16384 \
    --num_frames 32 \
    --fps 2.0 \
    --warmup_ratio 0.03 \
    --save_steps 10 \
    --save_total_limit 30 \
    --bf16 \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2.json"

echo "Done. SFT checkpoint -> $OUTPUT_DIR"
echo "下一步: bash scripts/submit_opd_from_sft.sh  (OPD GRPO+JSD 从这个checkpoint热启动)"
