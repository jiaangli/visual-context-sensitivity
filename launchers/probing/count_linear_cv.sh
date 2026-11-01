#!/usr/bin/env bash
# Linear count probe with 5-fold CV for one model.
#
# Usage:
#   MODEL_INDEX=0 bash launchers/probing/count_linear_cv.sh   # 0: Qwen2.5-VL-3B, 1: Qwen3.5-4B, 2: Gemma-4-E2B
# Under a SLURM array job the index defaults to SLURM_ARRAY_TASK_ID.
#
# Overrides: DATA_FILE, EXPID_PREFIX, RUN_TIMESTAMP, PROJECT_ROOT
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}"
cd "$project_root"

models=(
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen3.5-4B"
    "google/gemma-4-E2B-it"
)
aliases=(
    "qwen25vl3b"
    "qwen35_4b"
    "gemma4_e2b"
)
batch_sizes=(8 8 8)
worker_counts=(4 4 4)
attention_modes=(sdpa sdpa sdpa)

task_index="${MODEL_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
if (( task_index < 0 || task_index >= ${#models[@]} )); then
    echo "ERROR: MODEL_INDEX must be between 0 and $(( ${#models[@]} - 1 ))" >&2
    exit 2
fi

data_file="${DATA_FILE:-dataset/probing/count_cv_unique_generic/count_cv_unique_all.jsonl}"
expid_prefix="${EXPID_PREFIX:-count-linear-cv-generic}"
if [ ! -s "$data_file" ]; then
    echo "ERROR: missing CV dataset: ${data_file}" >&2
    echo "Run: python -m scripts.probing.build_count_probe_cv_dataset --generic-question --output-dir dataset/probing/count_cv_unique_generic" >&2
    exit 1
fi

source ./.venv/bin/activate

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${project_root}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

alias_name="${aliases[$task_index]}"
run_timestamp="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

python -m app.probe.count_last_token_linear_cv \
    --model-name-or-path "${models[$task_index]}" \
    --data-file "$data_file" \
    --image-root dataset \
    --conditions image no_image shuffled_image \
    --folds 5 \
    --c 1.0 \
    --max-iter 5000 \
    --tolerance 1e-4 \
    --bootstrap-samples 2000 \
    --batch-size "${batch_sizes[$task_index]}" \
    --no-image-batch-size 32 \
    --num-workers "${worker_counts[$task_index]}" \
    --attn-implementation "${attention_modes[$task_index]}" \
    --max-length 512 \
    --seed 42 \
    --expid "${expid_prefix}-${alias_name}" \
    --eval_category find_layers \
    --output_root results/eval \
    --timestamp "$run_timestamp"