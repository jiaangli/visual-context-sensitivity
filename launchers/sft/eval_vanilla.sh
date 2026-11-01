#!/usr/bin/env bash
# Evaluate the base model without a LoRA adapter.
#
# Usage:
#   bash launchers/sft/eval_vanilla.sh
#
# Override any value through the environment, for example:
#   MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct ALIAS=qwen25vl7b SEED=42 \
#       bash launchers/sft/eval_vanilla.sh
#   USE_IMAGE=False bash launchers/sft/eval_vanilla.sh  # text-only
#   GPU_ID=1 bash launchers/sft/eval_vanilla.sh         # run on GPU 1 (default: GPU 0)
#
# DATASETS defaults to the five validation files. For the test split:
#   DATASETS="dataset/whatifvis/color_test.jsonl,..." \
#   OUTPUT_NAME="qwen25vl3b_seed42_vanilla_instruction_five_test.json" \
#       bash launchers/sft/eval_vanilla.sh


set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd -- "$script_dir/../.." && pwd)}"
cd "$project_root"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${project_root}"
export TOKENIZERS_PARALLELISM=FALSE
# One GPU per run; GPU_ID (not an inherited CUDA_VISIBLE_DEVICES) picks it.
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
ALIAS="${ALIAS:-qwen25vl3b}"
SEED="${SEED:-42}"
USE_IMAGE="${USE_IMAGE:-True}"
IMAGE_ROOT="${IMAGE_ROOT:-dataset/whatifvis}"

# DATASETS="${DATASETS:-dataset/whatifvis/color_test.jsonl,dataset/whatifvis/count_test.jsonl,dataset/whatifvis/size_test.jsonl,dataset/whatifvis/spatial_temporal_test.jsonl,dataset/whatifvis/weight_test.jsonl}"
DATASETS="${DATASETS:-dataset/whatifvis/color_val.jsonl,dataset/whatifvis/count_val.jsonl,dataset/whatifvis/size_val.jsonl,dataset/whatifvis/spatial_temporal_val.jsonl,dataset/whatifvis/weight_val.jsonl}"
EXPID="${EXPID:-vanilla}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_NAME="${OUTPUT_NAME:-${ALIAS}_seed${SEED}_vanilla_instruction_five_val.json}"

# echo $PYTHONPATH

python -m evals.sft.inference \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_paths "$DATASETS" \
    --image_root "$IMAGE_ROOT" \
    --use_image "$USE_IMAGE" \
    --use_instruction True \
    --batch_size "${BATCH_SIZE:-8}" \
    --max_length "${MAX_LENGTH:-512}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
    --seed "$SEED" \
    --eval_category sft \
    --expid "$EXPID" \
    --timestamp "$RUN_TIMESTAMP" \
    --output_name "$OUTPUT_NAME"

echo "Results: results/eval/sft/${EXPID}/${RUN_TIMESTAMP}/${OUTPUT_NAME}"
