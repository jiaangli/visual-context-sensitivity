#!/usr/bin/env bash
# Search the important layer range for one SFT adapter and render its trace.
#
# Usage:
#   MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct \
#   ALIAS=qwen25vl3b \
#   CHECKPOINT=results/eval/sft/qwen25vl3b_sft/<timestamp>/checkpoints/<adapter> \
#       bash launchers/interp/find_layers.sh
#
# For text-only search add USE_IMAGE=False. Run once per TASK_DIRECTION to
# cover both directions.
#
# Overrides: SEED, BATCH_SIZE, TASK_DIRECTION, USE_IMAGE, RETENTION_THRESHOLD,
#            RETENTION_TAG, PLOT_EXTENSION, HIDE_ONSET, EXPID, RUN_TIMESTAMP,
#            OUTPUT_NAME, DATASET_PATH, PROJECT_ROOT
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the base model name or path}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the selected SFT adapter directory}"
ALIAS="${ALIAS:-$(basename -- "$MODEL_NAME")}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TASK_DIRECTION="${TASK_DIRECTION:-prior_to_context}"
USE_IMAGE="${USE_IMAGE:-True}"
RETENTION_THRESHOLD="${RETENTION_THRESHOLD:-0.8}"
RETENTION_TAG="${RETENTION_TAG:-080}"
PLOT_EXTENSION="${PLOT_EXTENSION:-png}"
HIDE_ONSET="${HIDE_ONSET:-False}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET_PATH="${DATASET_PATH:-dataset/whatifvis/spatial_temporal_val.jsonl}"
direction_tag="${TASK_DIRECTION//_/-}"

case "$USE_IMAGE" in
    True)
        default_expid="${ALIAS}-sft-seed${SEED}-${direction_tag}${RETENTION_TAG}"
        default_output_name="${ALIAS}_sft_seed${SEED}_${TASK_DIRECTION}_full_answer_normalized_retention${RETENTION_TAG}.json"
        ;;
    False)
        default_expid="${ALIAS}-text-statements-sft-seed${SEED}-${direction_tag}${RETENTION_TAG}"
        default_output_name="${ALIAS}_sft_seed${SEED}_${TASK_DIRECTION}_text_full_answer_normalized_retentions${RETENTION_TAG}.json"
        ;;
    *)
        echo "ERROR: USE_IMAGE must be exactly True or False" >&2
        exit 2
        ;;
esac
EXPID="${EXPID:-$default_expid}"
OUTPUT_NAME="${OUTPUT_NAME:-$default_output_name}"

if [ ! -f "${CHECKPOINT}/adapter_model.safetensors" ]; then
    echo "ERROR: selected SFT checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

run_dir="${project_root}/results/eval/find_layers/${EXPID}/${RUN_TIMESTAMP}"
trace_path="${run_dir}/search_trace/trace.json"
plot_path="${run_dir}/search_trace/full_answer_normalized.${PLOT_EXTENSION}"
if [ -e "${run_dir}/${OUTPUT_NAME}" ]; then
    echo "ERROR: refusing to overwrite existing layer search: ${run_dir}/${OUTPUT_NAME}" >&2
    exit 1
fi

python -m evals.interp.find_range \
    --model_name_or_path "$MODEL_NAME" \
    --lora_path "$CHECKPOINT" \
    --dataset_path "$DATASET_PATH" \
    --image_root dataset/whatifvis \
    --task_direction "$TASK_DIRECTION" \
    --use_image "$USE_IMAGE" \
    --batch_size "$BATCH_SIZE" \
    --require_distinct_answer_tokens True \
    --search_objective full_answer \
    --search_retention_threshold "$RETENTION_THRESHOLD" \
    --search_max_length 512 \
    --use_cache False \
    --seed "$SEED" \
    --eval_category find_layers \
    --expid "$EXPID" \
    --output_name "$OUTPUT_NAME" \
    --timestamp "$RUN_TIMESTAMP"
    

if [ ! -s "$trace_path" ]; then
    echo "ERROR: search completed without reusable trace: ${trace_path}" >&2
    exit 1
fi

plot_args=("$trace_path" --output-file "$plot_path")
if [ "$HIDE_ONSET" = "True" ]; then
    plot_args+=(--hide-onset)
fi
python -m evals.interp.plot_search_trace "${plot_args[@]}"

echo "Results: ${run_dir}/${OUTPUT_NAME}"
echo "Plot: ${plot_path}"
