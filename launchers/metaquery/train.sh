#!/usr/bin/env bash
# Train a MetaQuery image-reconstruction model (MLLM + Sana) on ImageNet-1k.
#
# Usage:
#   bash launchers/metaquery/train.sh
#
# ImageNet WebDataset shards are read from <data_dir>/imagenet-1k-wds/imagenet1k-train-*.tar
# (data_dir defaults to ./dataset/hf_data; override with --data_dir).
#
# Override any value through the environment, for example:
#   CONFIG_FILE=metaquery/qwen3p5vl_4b_sana_imagenet.yaml bash launchers/metaquery/train.sh
#   NPROC=1 CUDA_VISIBLE_DEVICES=0 bash launchers/metaquery/train.sh
# Extra arguments are passed through to app/imagegen/metaquery/train.py.
#
# Overrides: CONFIG_FILE, NPROC, MASTER_PORT, OUTPUT_DIR, CUDA_VISIBLE_DEVICES,
#            WANDB_MODE, NCCL_DEBUG, PROJECT_ROOT
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd -- "$script_dir/../.." && pwd)}"
cd "$project_root"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${project_root}"
export NUMEXPR_MAX_THREADS=16
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Config path is relative to configs/ (absolute paths also work).
# Note: values in the YAML override the matching command-line arguments.
CONFIG_FILE="${CONFIG_FILE:-metaquery/qwen2p5vl_3b_sana_imagenet.yaml}"
NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-12039}"

torchrun --nnodes=1 \
    --nproc-per-node="$NPROC" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="localhost:${MASTER_PORT}" \
    app/imagegen/metaquery/train.py \
    --config_file "$CONFIG_FILE" \
    --base_dir ./ \
    --output_dir "${OUTPUT_DIR:-./checkpoints/metaquery_qwen2p5}" \
    --optim adamw_8bit \
    "$@"
