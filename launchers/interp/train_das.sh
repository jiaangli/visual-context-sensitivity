#!/usr/bin/env bash
# Train one rank-one full-answer DAS projection for one model/layer.
#
# Usage:
# MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct \
# ALIAS=qwen25vl3b \
# CHECKPOINT=results/eval/sft/qwen25vl3b_sft/<timestamp>/checkpoints/<adapter> \
# LAYER=12 \
#     bash launchers/interp/train_das.sh
#
# For text-only training add USE_IMAGE=False. LAYER is the target layer; for a
# find_layers trace it can be read with:
#   python -c 'import json,sys; from src.interp.layer_selection import resolve_target_layer; print(resolve_target_layer(json.load(open(sys.argv[1]))))' <trace.json>
#
# Overrides: SEED, USE_IMAGE, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
#            FILTER_CRITERION, TRAIN_DATA, VAL_DATA, IMAGE_ROOT,
#            FILTER_CACHE_DIR, EXPID, RUN_TIMESTAMP, OUTPUT_NAME, DAS_RANK,
#            DAS_EPOCHS, DAS_LR, GRAD_ACC_STEPS, WARMUP_RATIO, MAX_LENGTH,
#            MAX_SAMPLES, PROJECT_ROOT
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd -- "$script_dir/../.." && pwd)}"
cd "$project_root"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${project_root}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=8
# One GPU per run; GPU_ID (not an inherited CUDA_VISIBLE_DEVICES) picks it.
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the base model name or path}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the selected SFT adapter directory}"
ALIAS="${ALIAS:-$(basename -- "$MODEL_NAME")}"
LAYER="${LAYER:?Set LAYER to the target layer}"
SEED="${SEED:-42}"
USE_IMAGE="${USE_IMAGE:-True}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
FILTER_CRITERION="${FILTER_CRITERION:-first_token_argmax}"
TRAIN_DATA="${TRAIN_DATA:-dataset/whatifvis/spatial_temporal_train.jsonl}"
VAL_DATA="${VAL_DATA:-dataset/whatifvis/spatial_temporal_val.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-dataset/whatifvis}"
DAS_RANK="${DAS_RANK:-1}"
DAS_EPOCHS="${DAS_EPOCHS:-1}"
DAS_LR="${DAS_LR:-1e-3}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
EXPID="${EXPID:-das-${ALIAS}-layer${LAYER}}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

case "$USE_IMAGE" in
    True)
        default_output_name="${ALIAS}_seed${SEED}_layer${LAYER}_das_full_answer_training.json"
        ;;
    False)
        default_output_name="${ALIAS}_seed${SEED}_layer${LAYER}_text_das_full_answer_training.json"
        ;;
    *)
        echo "ERROR: USE_IMAGE must be exactly True or False" >&2
        exit 2
        ;;
esac
OUTPUT_NAME="${OUTPUT_NAME:-$default_output_name}"

if [ ! -f "${CHECKPOINT}/adapter_model.safetensors" ]; then
    echo "ERROR: selected SFT checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

# run_dir="${project_root}/results/eval/steering/${EXPID}/${RUN_TIMESTAMP}"
run_dir="${project_root}/results/eval/steering/${EXPID}"
result_path="${run_dir}/${RUN_TIMESTAMP}/${OUTPUT_NAME}"
projection_path="${run_dir}/${RUN_TIMESTAMP}/projection_layer_${LAYER}/model.safetensors"
filter_cache_dir="${FILTER_CACHE_DIR:-${run_dir}/filter_cache}"
if [ -e "$result_path" ] || [ -e "$projection_path" ]; then
    echo "ERROR: refusing to overwrite an existing training artifact in ${run_dir}" >&2
    exit 1
fi

python -m app.interp.subspace_train \
    --model_name_or_path "$MODEL_NAME" \
    --lora_path "$CHECKPOINT" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --image_root "$IMAGE_ROOT" \
    --use_image "$USE_IMAGE" \
    --target_layers "$LAYER" \
    --skip_layer_search True \
    --das_rank "$DAS_RANK" \
    --das_epochs "$DAS_EPOCHS" \
    --das_lr "$DAS_LR" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --grad_acc_steps "$GRAD_ACC_STEPS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --objective full_answer \
    --filter_criterion "$FILTER_CRITERION" \
    --max_length "$MAX_LENGTH" \
    --filter_batch_size "$EVAL_BATCH_SIZE" \
    --max_samples "$MAX_SAMPLES" \
    --cache_dir "$filter_cache_dir" \
    --use_cache True \
    --seed "$SEED" \
    --eval_category steering \
    --expid "$EXPID" \
    --output_name "$OUTPUT_NAME" \
    --timestamp "$RUN_TIMESTAMP" 

if [ ! -s "$result_path" ] || [ ! -s "$projection_path" ]; then
    echo "ERROR: training completed without the expected result/projection" >&2
    exit 1
fi

echo "Results: ${result_path}"
echo "Projection: ${run_dir}/${RUN_TIMESTAMP}/projection_layer_${LAYER}"


