#!/usr/bin/env bash
# Sweep context/prior DAS coefficients for one projection on the five validation sets.
#
# Usage:
#   MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct \
#   ALIAS=qwen25vl3b \
#   PROJECTION=results/eval/steering/das-qwen25vl3b-layer12/<timestamp>/projection_layer_12 \
#   LAYER=12 \
#   BASE_REFERENCE=results/eval/sft/vanilla/<timestamp>/qwen25vl3b_seed42_vanilla_instruction_five_val.json \
#   SFT_REFERENCE=results/eval/sft/sft/<timestamp>/qwen25vl3b_seed42_sft_instruction_five_val.json \
#       bash launchers/interp/sweep_values.sh
#
# BASE_REFERENCE / SFT_REFERENCE are the five-val result JSONs written by
# launchers/sft/eval_vanilla.sh and launchers/sft/eval_sft.sh.
# SFT_REFERENCE=none skips the SFT reference (closest-to-SFT selection is then omitted).
# For text-only steering add USE_IMAGE=False.
#
# Overrides: SEED, CONTEXT_VALUES, PRIOR_VALUES, USE_IMAGE, DATASETS,
#            MAX_ROWS_PER_DATASET, RESUME_CHECKPOINT_PATH, EXPID,
#            RUN_TIMESTAMP, OUTPUT_NAME, BATCH_SIZE, MAX_LENGTH,
#            MAX_NEW_TOKENS, PROJECT_ROOT
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
SEED="${SEED:-42}"
USE_IMAGE="${USE_IMAGE:-True}"
# CONTEXT_VALUES="${CONTEXT_VALUES:--10,-5,0,5,10}"
CONTEXT_VALUES="${CONTEXT_VALUES:--68}"
PRIOR_VALUES="${PRIOR_VALUES:-0,5,10,15,20,25,30,35,40,45,50,55,60,65,70}"
MAX_ROWS_PER_DATASET="${MAX_ROWS_PER_DATASET:-65}"
EXPID="${EXPID:-interp-value-sweep}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASETS="${DATASETS:-dataset/whatifvis/color_val.jsonl,dataset/whatifvis/count_val.jsonl,dataset/whatifvis/size_val.jsonl,dataset/whatifvis/spatial_temporal_val.jsonl,dataset/whatifvis/weight_val.jsonl}"
# DATASETS="${DATASETS:-dataset/whatifvis/spatial_temporal_val.jsonl}"
OUTPUT_NAME="${OUTPUT_NAME:-${ALIAS}_seed${SEED}_layer${LAYER}_prefix_once_val_sweep.json}"

case "$USE_IMAGE" in
    True|False) ;;
    *) echo "ERROR: USE_IMAGE must be exactly True or False" >&2; exit 2 ;;
esac

if [[ "$PROJECTION" = /* ]]; then
    projection_path="$PROJECTION"
else
    projection_path="${project_root}/${PROJECTION}"
fi

base_reference="${BASE_REFERENCE:?Set BASE_REFERENCE to the vanilla five-val result JSON}"
sft_reference="${SFT_REFERENCE:?Set SFT_REFERENCE to the SFT five-val result JSON, or to none}"
required_paths=("${projection_path}/model.safetensors" "$base_reference")
reference_args=(--base_reference_path "$base_reference")
if [ "$sft_reference" != "none" ]; then
    required_paths+=("$sft_reference")
    reference_args+=(--sft_reference_path "$sft_reference")
fi
for required_path in "${required_paths[@]}"; do
    if [ ! -f "$required_path" ]; then
        echo "ERROR: required artifact not found: ${required_path}" >&2
        exit 1
    fi
done

row_limit_args=()
if [ "$MAX_ROWS_PER_DATASET" != "all" ]; then
    if ! [[ "$MAX_ROWS_PER_DATASET" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: MAX_ROWS_PER_DATASET must be a positive integer or all" >&2
        exit 2
    fi
    row_limit_args+=(--max_rows_per_dataset "$MAX_ROWS_PER_DATASET")
fi

resume_args=()
if [ -n "${RESUME_CHECKPOINT_PATH:-}" ]; then
    resume_args+=(--resume_checkpoint_path "$RESUME_CHECKPOINT_PATH")
fi

run_dir="${project_root}/results/eval/steering/${EXPID}/${RUN_TIMESTAMP}"
if [ -e "${run_dir}/${OUTPUT_NAME}" ]; then
    echo "ERROR: refusing to overwrite existing sweep result: ${run_dir}/${OUTPUT_NAME}" >&2
    exit 1
fi

python -m evals.interp.subspace_value_sweep \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path "$DATASETS" \
    --image_root dataset/whatifvis \
    --projection_path "$projection_path" \
    --steering_layer "$LAYER" \
    --context_values="$CONTEXT_VALUES" \
    --prior_values="$PRIOR_VALUES" \
    --steering_hook_scope response_prefix \
    "${reference_args[@]}" \
    --use_image "$USE_IMAGE" \
    --use_instruction False \
    --batch_size "${BATCH_SIZE:-16}" \
    --max_length "${MAX_LENGTH:-512}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-16}" \
    --seed "$SEED" \
    --eval_category steering \
    --expid "$EXPID" \
    --timestamp "$RUN_TIMESTAMP" \
    --output_name "$OUTPUT_NAME" \
    "${row_limit_args[@]}" \
    "${resume_args[@]}"

echo "Results: ${run_dir}/${OUTPUT_NAME}"
