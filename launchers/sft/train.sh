#!/usr/bin/env bash
# Train one SFT LoRA adapter.
#
# Usage:
#   bash launchers/sft/train.sh
#
# Override any value through the environment, for example:
#   MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct ALIAS=qwen25vl7b SEED=42 \
#       bash launchers/sft/train.sh
#   USE_IMAGE=False bash launchers/sft/train.sh   # text-only training
#   GPU_ID=1 bash launchers/sft/train.sh          # train on GPU 1 (default: GPU 0)
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

# Model and training settings.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
ALIAS="${ALIAS:-qwen25vl3b}"
SEED="${SEED:-42}"
LORA_R="${LORA_R:-16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
USE_IMAGE="${USE_IMAGE:-True}"

# Data settings.
TRAIN_DATASET="${TRAIN_DATASET:-dataset/whatifvis/spatial_temporal_train.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-dataset/whatifvis}"
VAL_DATASETS="${VAL_DATASETS:-dataset/whatifvis/color_val.jsonl,dataset/whatifvis/count_val.jsonl,dataset/whatifvis/size_val.jsonl,dataset/whatifvis/spatial_temporal_val.jsonl,dataset/whatifvis/weight_val.jsonl}"
# VAL_DATASETS="${VAL_DATASETS:-dataset/whatifvis/spatial_temporal_val.jsonl}"
VAL_IMAGE_ROOT="${VAL_IMAGE_ROOT:-$IMAGE_ROOT}"
TEST_DATASETS="${TEST_DATASETS:-dataset/whatifvis/color_test.jsonl,dataset/whatifvis/count_test.jsonl,dataset/whatifvis/size_test.jsonl,dataset/whatifvis/spatial_temporal_test.jsonl,dataset/whatifvis/weight_test.jsonl}"
TEST_IMAGE_ROOT="${TEST_IMAGE_ROOT:-$IMAGE_ROOT}"

# Output settings.
EXPID="${EXPID:-${ALIAS}_sft}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
CHECKPOINT_NAME="${ALIAS}_spatial&temporal_use-image_${USE_IMAGE}_lora_${LORA_R}_seed_${SEED}"
OUTPUT_DIR="${OUTPUT_DIR:-results/eval/sft/${EXPID}/${RUN_TIMESTAMP}/checkpoints/${CHECKPOINT_NAME}}"
OUTPUT_NAME="${OUTPUT_NAME:-${CHECKPOINT_NAME}_training.json}"

# Checkpoint selection. Loss-type metrics are minimised, everything else maximised.
SELECTION_METRIC="${SELECTION_METRIC:-eval_pair_accuracy}"
case "$SELECTION_METRIC" in
    *loss*) default_greater_is_better=False ;;
    *) default_greater_is_better=True ;;
esac
GREATER_IS_BETTER="${GREATER_IS_BETTER:-$default_greater_is_better}"

echo "Training ${MODEL_NAME} (${ALIAS}), seed=${SEED}, LoRA rank=${LORA_R}"
echo "Checkpoint: ${OUTPUT_DIR}"

python -m app.sft.train \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_paths "$TRAIN_DATASET" \
    --image_root "$IMAGE_ROOT" \
    --validation_dataset_path "$VAL_DATASETS" \
    --validation_image_root "$VAL_IMAGE_ROOT" \
    --validation_max_examples_per_dataset "${VAL_MAX_EXAMPLES_PER_DATASET:-65}" \
    --test_dataset_paths "$TEST_DATASETS" \
    --test_image_root "$TEST_IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --eval_category sft \
    --expid "$EXPID" \
    --timestamp "$RUN_TIMESTAMP" \
    --output_name "$OUTPUT_NAME" \
    --run_name "${ALIAS}_seed_${SEED}_${RUN_TIMESTAMP}" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 5e-6}' \
    --warmup_ratio 0.03 \
    --weight_decay 0.001 \
    --optim adamw_torch_fused \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-False}" \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model "$SELECTION_METRIC" \
    --greater_is_better "$GREATER_IS_BETTER" \
    --report_to "${REPORT_TO:-none}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --logging_steps 10 \
    --max_grad_norm 1 \
    --lora_r "$LORA_R" \
    --lora_alpha "$((LORA_R * 2))" \
    --normalize_stop_loss_by_token_count "${NORMALIZE_STOP_LOSS_BY_TOKEN_COUNT:-True}" \
    --stop_loss_weight "${STOP_LOSS_WEIGHT:-1}" \
    --use_image "$USE_IMAGE" \
    --seed "$SEED"
