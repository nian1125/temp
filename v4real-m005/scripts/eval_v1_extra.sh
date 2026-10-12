#!/bin/bash
# v1 额外测评: checkpoint-50 + 根目录最终模型
# 用法: bash scripts/eval_v1_extra.sh
set -e

TIMELENS_ROOT="/primus_xpfs_root/zhangyajie/czy/timelens/TimeLens"
OPD_OUTPUT="/primus_xpfs_root/zhangyajie/czy/aaacode/Temporal-ZwZ-Qwen3/output/opd-grpo"
V1_DIR="$OPD_OUTPUT/opd-numberit+red_border-kl1.0-lr1e-6-20260617-1118"

cd "$TIMELENS_ROOT"
export PYTHONPATH="./:$PYTHONPATH"
pip install decord pillow qwen-vl-utils nncore 2>/dev/null

export min_tokens=64
export total_tokens=14336
export FPS=2
export datasets="charades-timelens"

RESULTS_FILE="$OPD_OUTPUT/eval_v1_extra_results.txt"
echo "# v1 Extra Evaluation Results ($(date))" > "$RESULTS_FILE"

# 1. 测评 checkpoint-50
tag="v1-step50"
ckpt="$V1_DIR/checkpoint-50"
if [ -d "$ckpt" ]; then
    echo ""
    echo "=========================================="
    echo "Evaluating: $tag"
    echo "  Path: $ckpt"
    echo "=========================================="
    export model_path="$ckpt"
    export pred_path="$OPD_OUTPUT/eval_preds"
    bash scripts/eval_timelens_bench.sh 2>&1 | tee /tmp/eval_${tag}.log
    echo "" >> "$RESULTS_FILE"
    echo "## $tag" >> "$RESULTS_FILE"
    echo "Path: $ckpt" >> "$RESULTS_FILE"
    grep -E "IOU|mIOU|Recall|R1" /tmp/eval_${tag}.log >> "$RESULTS_FILE" 2>/dev/null || echo "metrics not captured" >> "$RESULTS_FILE"
else
    echo "[SKIP] $tag: not found"
fi

# 2. 测评根目录最终模型 (可能是 step 300)
tag="v1-final"
ckpt="$V1_DIR"
if [ -f "$ckpt/model-00001-of-00004.safetensors" ]; then
    echo ""
    echo "=========================================="
    echo "Evaluating: $tag (root dir model)"
    echo "  Path: $ckpt"
    echo "=========================================="

    # 打印实际步数
    python3 -c "import json; s=json.load(open('$ckpt/trainer_state.json')); print(f'  Root model is at step {s[\"global_step\"]}')" 2>/dev/null || true

    export model_path="$ckpt"
    export pred_path="$OPD_OUTPUT/eval_preds"
    bash scripts/eval_timelens_bench.sh 2>&1 | tee /tmp/eval_${tag}.log
    echo "" >> "$RESULTS_FILE"
    echo "## $tag" >> "$RESULTS_FILE"
    echo "Path: $ckpt" >> "$RESULTS_FILE"
    python3 -c "import json; s=json.load(open('$ckpt/trainer_state.json')); print(f'Actual step: {s[\"global_step\"]}')" >> "$RESULTS_FILE" 2>/dev/null || true
    grep -E "IOU|mIOU|Recall|R1" /tmp/eval_${tag}.log >> "$RESULTS_FILE" 2>/dev/null || echo "metrics not captured" >> "$RESULTS_FILE"
else
    echo "[SKIP] $tag: no model files in root dir"
fi

echo ""
echo "=========================================="
echo "v1 extra evaluation complete."
echo "=========================================="
cat "$RESULTS_FILE"
