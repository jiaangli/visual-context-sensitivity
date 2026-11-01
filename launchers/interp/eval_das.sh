#!/usr/bin/env bash
# Evaluate one frozen DAS steering condition on the five test sets.
#
# Usage:
#   MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct \
#   ALIAS=qwen25vl3b \
#   PROJECTION=results/eval/steering/das-qwen25vl3b-layer12/<timestamp>/projection_layer_12 \
#   LAYER=12 \
#   PRIOR_VALUE=-1 \
#   CONTEXT_VALUE=1 \
#       bash launchers/interp/eval_das.sh
#
# For text-only steering add USE_IMAGE=False.
#
# Overrides: SEED, USE_IMAGE, DATASETS, EXPID, RUN_TIMESTAMP, OUTPUT_NAME,
#            BATCH_SIZE, MAX_LENGTH, MAX_NEW_TOKENS, PROJECT_ROOT
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
# One GPU per run; GPU_ID (not an inherited CUDA_VISIBLE_DEVICES) picks it.
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the base model name or path}"
ALIAS="${ALIAS:-$(basename -- "$MODEL_NAME")}"
PROJECTION="${PROJECTION:?Set PROJECTION to the trained projection directory}"
LAYER="${LAYER:?Set LAYER to the steering layer}"
PRIOR_VALUE="${PRIOR_VALUE:?Set PRIOR_VALUE to the frozen prior coefficient}"
CONTEXT_VALUE="${CONTEXT_VALUE:?Set CONTEXT_VALUE to the frozen context coefficient}"
SEED="${SEED:-42}"
USE_IMAGE="${USE_IMAGE:-True}"
EXPID="${EXPID:-interp-eval}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASETS="${DATASETS:-dataset/whatifvis/color_test.jsonl,dataset/whatifvis/count_test.jsonl,dataset/whatifvis/size_test.jsonl,dataset/whatifvis/spatial_temporal_test.jsonl,dataset/whatifvis/weight_test.jsonl}"

case "$USE_IMAGE" in
    True)
        default_output_name="${ALIAS}_seed${SEED}_layer${LAYER}_p${PRIOR_VALUE}_c${CONTEXT_VALUE}_five_test.json"
        ;;
    False)
        default_output_name="${ALIAS}_base_seed${SEED}_layer${LAYER}_p${PRIOR_VALUE}_c${CONTEXT_VALUE}_text_five_test.json"
        ;;
    *)
        echo "ERROR: USE_IMAGE must be exactly True or False" >&2
        exit 2
        ;;
esac
OUTPUT_NAME="${OUTPUT_NAME:-$default_output_name}"

if [[ "$PROJECTION" = /* ]]; then
    projection_path="$PROJECTION"
else
    projection_path="${project_root}/${PROJECTION}"
fi
if [ ! -s "${projection_path}/model.safetensors" ]; then
    echo "ERROR: projection not found: ${projection_path}/model.safetensors" >&2
    exit 1
fi

run_dir="${project_root}/results/eval/steering/${EXPID}/${RUN_TIMESTAMP}"
if [ -e "${run_dir}/${OUTPUT_NAME}" ]; then
    echo "ERROR: refusing to overwrite existing evaluation result: ${run_dir}/${OUTPUT_NAME}" >&2
    exit 1
fi

python -m evals.interp.subspace_inference \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_paths "$DATASETS" \
    --image_root dataset/whatifvis \
    --projection_path "$projection_path" \
    --steering_layer "$LAYER" \
    --steering_prior_value "$PRIOR_VALUE" \
    --steering_context_value "$CONTEXT_VALUE" \
    --steering_hook_scope response_prefix \
    --use_image "$USE_IMAGE" \
    --use_instruction False \
    --batch_size "${BATCH_SIZE:-8}" \
    --max_length "${MAX_LENGTH:-512}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
    --seed "$SEED" \
    --eval_category steering \
    --expid "$EXPID" \
    --timestamp "$RUN_TIMESTAMP" \
    --output_name "$OUTPUT_NAME"

echo "Results: ${run_dir}/${OUTPUT_NAME}"
