#!/bin/bash
# ============================================================
# Student rollout 打分: 对 6838 条各采样8次, 算 reward_std (GRPO信号)
# 在 base/student 模型上跑。输出每条 reward_mean/reward_std/8个iou。
# reward_std>0 = student时对时错 = GRPO有梯度信号。
# ============================================================
set -e
cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3
pip install decord pillow qwen-vl-utils nncore imageio imageio-ffmpeg 2>/dev/null
pip install /primus_xpfs_root/zhangyajie/czy/DeepEyes/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct"
TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
ANNO="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/teacher_scores_quick/filtered_10k_gt0.3.jsonl"
OUT_DIR="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/student_rollout_scores"
mkdir -p "$OUT_DIR"

CHUNKS=8
GPULIST=(0 1 2 3 4 5 6 7)
for IDX in "${!GPULIST[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python scripts/student_rollout_scores.py \
        --anno_path "$ANNO" \
        --video_dir "$TIMELENS_ROOT" \
        --model_path "$MODEL_PATH" \
        --output_path "$OUT_DIR/scores" \
        --num_generations 8 \
        --temperature 1.0 \
        --chunk "$CHUNKS" \
        --index "$IDX" &
done
echo "Waiting for $CHUNKS shards..."
wait
cat "$OUT_DIR"/scores_*.jsonl > "$OUT_DIR/student_scores_all.jsonl"
echo "Done. Aggregated: $(wc -l < "$OUT_DIR/student_scores_all.jsonl") samples -> $OUT_DIR/student_scores_all.jsonl"
echo "把 student_scores_all.jsonl 传回OSS, 我来做三信号(teacher_tiou × student_iou × reward_std)联合筛选"
