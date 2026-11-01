#!/usr/bin/env bash
# Generate the five-task MetaQuery reconstruction suite and evaluate
# reconstruction metrics for the three MetaQuery checkpoints.
#
# Usage:
#   QWEN25VL3B_CHECKPOINT=/path/to/checkpoint-10000 \
#   QWEN35_4B_CHECKPOINT=/path/to/checkpoint-10000 \
#   GEMMA4_E2B_CHECKPOINT=/path/to/checkpoint-20000 \
#       bash launchers/metaquery/run_five_test_reconstruction.sh
#
# Overrides: RUN_TIMESTAMP, SUITE_DIR, GENERATION_BATCH_SIZE,
#            NUM_INFERENCE_STEPS, GUIDANCE_SCALE, IMAGE_GUIDANCE_SCALE,
#            RECONSTRUCTION_SEED, MAX_IMAGES_PER_TASK, METRIC_BATCH_SIZE,
#            PROJECT_ROOT
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd -- "$script_dir/../.." && pwd)}"
cd "$project_root"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${project_root}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

qwen25vl3b_checkpoint="${QWEN25VL3B_CHECKPOINT:?Set QWEN25VL3B_CHECKPOINT to the Qwen2.5-VL-3B MetaQuery checkpoint}"
qwen35_4b_checkpoint="${QWEN35_4B_CHECKPOINT:?Set QWEN35_4B_CHECKPOINT to the Qwen3.5-4B MetaQuery checkpoint}"
gemma4_e2b_checkpoint="${GEMMA4_E2B_CHECKPOINT:?Set GEMMA4_E2B_CHECKPOINT to the Gemma-4-E2B MetaQuery checkpoint}"

run_timestamp="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
suite_dir="${SUITE_DIR:-${project_root}/results/eval/metaquery/reconstruction-five-test/${run_timestamp}}"

model_aliases=(qwen25vl3b qwen35_4b gemma4_e2b)
model_checkpoints=(
    "$qwen25vl3b_checkpoint"
    "$qwen35_4b_checkpoint"
    "$gemma4_e2b_checkpoint"
)

for index in "${!model_aliases[@]}"; do
    checkpoint="${model_checkpoints[$index]}"
    if [ ! -s "${checkpoint}/model.safetensors" ] || [ ! -s "${checkpoint}/config.json" ]; then
        echo "ERROR: incomplete MetaQuery checkpoint: ${checkpoint}" >&2
        exit 1
    fi
done

if [ -e "$suite_dir" ]; then
    echo "ERROR: refusing to reuse existing suite directory: ${suite_dir}" >&2
    exit 1
fi
mkdir -p "$suite_dir"

for index in "${!model_aliases[@]}"; do
    model_alias="${model_aliases[$index]}"
    checkpoint="${model_checkpoints[$index]}"
    echo "Generating reconstructions for ${model_alias}"
    python -m evals.imagegen.metaquery.generate_reconstruction_suite \
        --suite-dir "$suite_dir" \
        --model-alias "$model_alias" \
        --checkpoint-path "$checkpoint" \
        --dataset-root dataset/whatifvis \
        --batch-size "${GENERATION_BATCH_SIZE:-1}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS:-30}" \
        --guidance-scale "${GUIDANCE_SCALE:-0}" \
        --image-guidance-scale "${IMAGE_GUIDANCE_SCALE:-1.5}" \
        --seed "${RECONSTRUCTION_SEED:-42}" \
        --color-correct \
        --max-images-per-task "${MAX_IMAGES_PER_TASK:-0}"
done

echo "Evaluating reconstruction metrics"
python -m evals.imagegen.metaquery.eval_reconstruction_suite \
    --suite-dir "$suite_dir" \
    --models "${model_aliases[@]}" \
    --factual-dir dataset/whatifvis/images/vlms_are_biased_factual \
    --batch-size "${METRIC_BATCH_SIZE:-16}"

echo "Suite: ${suite_dir}"
