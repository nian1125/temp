#!/bin/bash
# Gap Test: 验证 Teacher（GT标注帧）vs Student（原始视频）的零样本性能差距
# 用于决定方法路线：纯自蒸馏 or GRPO+蒸馏

cd /primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3

# ============================================================
# 环境检查
# ============================================================
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

pip install decord pillow qwen-vl-utils opencv-python-headless 2>/dev/null

# ============================================================
# 路径配置
# ============================================================
export TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
export HF_ENDPOINT=https://hf-mirror.com

MODEL_PATH="/primus_xpfs_root/zhangyajie/ckpt/Qwen3-VL-8B-Instruct"
ANNO_PATH="${TIMELENS_ROOT}/data/filter-data/FPS-2-maxframes-448_TOTALtokens-14336_MINtokens-64---20251209_223300/gemini_refined_data.jsonl"
VIDEO_DIR="${TIMELENS_ROOT}"
DATASET="filtered"
OUTPUT_DIR="results/teacher_gap_test"

# ============================================================
# 参数（可按需调整）
# ============================================================
FPS=2               # 与训练/推理保持一致
NUM_FRAMES=448      # 帧数上限（FPS=2 控制实际采样数）
MARK_MAX_FRAMES=32  # GT-free 标注策略的帧数上限（图片模式）
MAX_SAMPLES=200     # 测试样本数（200够判断趋势，500更稳）
STRATEGIES="clean,numberit+red_border"  # 对比：原始 vs GT标注
TEACHER_CACHE_DIR="results/teacher_video_cache"  # Teacher 视频缓存目录

# ============================================================
# 运行 Gap 测试
# ============================================================
echo "=========================================="
echo "Teacher Advantage Gap Test"
echo "=========================================="
echo "Model:      ${MODEL_PATH}"
echo "Dataset:    ${DATASET}"
echo "Benchmark:  ${ANNO_PATH}"
echo "Videos:     ${VIDEO_DIR}"
echo "Strategies: ${STRATEGIES}"
echo "FPS:        ${FPS}"
echo "Max Frames: ${NUM_FRAMES}"
echo "Mark Frames:${MARK_MAX_FRAMES}"
echo "Cache Dir:  ${TEACHER_CACHE_DIR}"
echo "Samples:    ${MAX_SAMPLES}"
echo "=========================================="

python scripts/zero_shot_test.py \
    --model_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --anno_path "${ANNO_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --strategies "${STRATEGIES}" \
    --fps ${FPS} \
    --num_frames ${NUM_FRAMES} \
    --mark_max_frames ${MARK_MAX_FRAMES} \
    --teacher_cache_dir ${TEACHER_CACHE_DIR} \
    --max_samples ${MAX_SAMPLES} \
    --output_dir "${OUTPUT_DIR}" \
    --save_samples 5

echo "=========================================="
echo "Done. Results: ${OUTPUT_DIR}/comparison.json"
echo "=========================================="
