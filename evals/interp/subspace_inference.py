"""Evaluate a learned DAS subspace with constant steering interventions.

The evaluator uses the same versioned metrics as SFT inference: strict
full-answer exact match and full-answer probability preference.
"""

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, HfArgumentParser

from evals.sft.inference import (
    METRIC_PROTOCOL,
    compute_macro_metrics,
    compute_metrics,
    describe_model_precision,
    describe_runtime,
    exact_match,
    log_metrics,
    normalize_answer,
)
from src.eval_generation import build_eval_generation_kwargs, describe_eval_generation
from src.eval_results import (
    DEFAULT_EVAL_ROOT,
    configure_run_logging,
    prepare_eval_run,
    write_json_result,
)
from src.interp.das import LowRankOrthogonalProjection
from src.interp.steering import BinaryHook
from src.model_utils import get_device, load_model_and_processor, setup_seed
from src.sft.data_process import CustomCollator, Task, TaskMixture
from src.sft.preference_eval import (
    build_probability_inputs,
    candidate_token_ids,
    sequence_logprobs,
)

load_dotenv()
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

STEERING_HOOK_SCOPES = {"response_prefix"}


def build_steering_metric_protocol(steering_hook_scope: str) -> dict[str, Any]:
    if steering_hook_scope not in STEERING_HOOK_SCOPES:
        raise ValueError(
            f"steering_hook_scope must be one of {sorted(STEERING_HOOK_SCOPES)}, got {steering_hook_scope!r}"
        )
    return {
        **METRIC_PROTOCOL,
        "version": "steering-eval-v5",
        "steering": {
            "value_convention": "signed_hook_value",
            "hook_scope": steering_hook_scope,
            "generation": (
                "absolute DAS projection-coordinate replacement once, at the final "
                "response-prefix token during the cached prefill pass"
            ),
            "probability": (
                "score every candidate-answer token after applying absolute DAS "
                "projection-coordinate replacement only at the final response-prefix token"
            ),
        },
    }


STEERING_METRIC_PROTOCOL = build_steering_metric_protocol("response_prefix")


@dataclass
class InferenceArguments:
    expid: str = field(
        default="steering-eval",
        metadata={"help": "Experiment id used to group related eval jobs"},
    )
    eval_category: str = field(
        default="steering",
        metadata={"help": "Experiment result category"},
    )
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Base model path"},
    )
    lora_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Optional LoRA adapter path"},
    )
    dataset_paths: str = field(
        default="dataset/whatifvis/spatial_temporal_test.jsonl",
        metadata={"help": "Comma-separated evaluation JSONL paths"},
    )
    image_root: str | None = field(
        default="dataset/whatifvis",
        metadata={"help": "Optional root to prepend to image paths"},
    )
    output_root: str = field(
        default=DEFAULT_EVAL_ROOT,
        metadata={"help": "Root directory for evaluation artifacts"},
    )
    timestamp: str | None = field(default=None)
    output_name: str = field(default="results.json")
    batch_size: int = field(default=4)
    max_length: int = field(default=512)
    max_new_tokens: int = field(
        default=16,
        metadata={"help": "Maximum generated tokens for exact-match evaluation"},
    )
    stop: int | None = field(default=None)
    steering_layer: str = field(
        default="0",
        metadata={"help": "Comma-separated 0-based layer indices"},
    )
    steering_prior_value: float = field(default=-50.0)
    steering_context_value: float = field(default=50.0)
    steering_hook_scope: str = field(
        default="response_prefix",
        metadata={
            "help": (
                "Intervention scope: response_prefix applies once at the final "
                "response-prefix token for generation and probability scoring"
            )
        },
    )
    projection_path: str = field(
        default="projections/qwen2p5vl_3b_layer_26",
        metadata={"help": "Path to a learned projection"},
    )
    use_image: bool = field(default=True)
    use_instruction: bool = field(
        default=False,
        metadata={"help": "Whether the neutral steering prompt includes intent instructions"},
    )
    seed: int = field(default=42)


def _autocast(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def response_prefix_position_mask(labels: torch.Tensor) -> torch.Tensor:
    """Select only the hidden state that predicts the first answer token."""
    if labels.ndim != 2:
        raise ValueError(f"Expected rank-2 labels, got shape {tuple(labels.shape)}")
    labeled = labels.ne(-100)
    if torch.any(~labeled.any(dim=1)):
        raise ValueError("Every row must contain at least one labeled answer token")
    first_answer_positions = labeled.to(torch.int64).argmax(dim=1)
    if torch.any(first_answer_positions == 0):
        raise ValueError("Every answer must be preceded by a response-prefix token")
    mask = torch.zeros_like(labels, dtype=torch.bool)
    rows = torch.arange(labels.shape[0], device=labels.device)
    mask[rows, first_answer_positions - 1] = True
    return mask


def _probability_scores(
    model,
    inputs: Mapping[str, torch.Tensor],
    candidates: Sequence[Sequence[int]],
    *,
    hooks: Sequence[BinaryHook],
    pad_token_id: int,
    max_length: int,
    device: str,
) -> tuple[list[float], list[float], list[int]]:
    candidate_inputs, labels = build_probability_inputs(
        inputs,
        candidates,
        pad_token_id=pad_token_id,
        max_length=max_length,
    )
    position_mask = response_prefix_position_mask(labels)
    for hook in hooks:
        hook.set_position_mask(position_mask)
    try:
        with _autocast(device):
            output = model(
                **candidate_inputs,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for hook in hooks:
            hook.set_position_mask(None)
    sums, means, counts = sequence_logprobs(output.logits, labels)
    return (
        sums.detach().cpu().tolist(),
        means.detach().cpu().tolist(),
        counts.detach().cpu().tolist(),
    )


def score_batch(
    model,
    inputs: dict[str, torch.Tensor],
    processor: AutoProcessor,
    context_answers: Sequence[str],
    prior_answers: Sequence[str],
    response_prefixes: Sequence[str],
    hooks: Sequence[BinaryHook],
    *,
    device: str,
    max_length: int,
    max_new_tokens: int,
) -> tuple[list[str], list[dict[str, float | int]]]:
    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    if "gemma" in processor.tokenizer.name_or_path.lower():
        inputs["token_type_ids"] = inputs["mm_token_type_ids"]

    input_ids = inputs["input_ids"]
    context_ids = candidate_token_ids(processor, context_answers, response_prefixes)
    prior_ids = candidate_token_ids(processor, prior_answers, response_prefixes)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer defines neither pad_token_id nor eos_token_id")

    generation_kwargs = build_eval_generation_kwargs(model, input_ids)
    with _autocast(device):
        output = model.generate(
            **inputs,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            use_cache=True,
            **generation_kwargs,
        )

    generated_tokens = output.sequences[:, input_ids.shape[1] :]
    decoded = processor.batch_decode(generated_tokens, skip_special_tokens=True)
    del output

    context_sums, context_means, context_counts = _probability_scores(
        model,
        inputs,
        context_ids,
        hooks=hooks,
        pad_token_id=pad_token_id,
        max_length=max_length,
        device=device,
    )
    prior_sums, prior_means, prior_counts = _probability_scores(
        model,
        inputs,
        prior_ids,
        hooks=hooks,
        pad_token_id=pad_token_id,
        max_length=max_length,
        device=device,
    )

    scores: list[dict[str, float | int]] = []
    for index in range(len(context_answers)):
        scores.append(
            {
                "probability_context_logprob_sum": context_sums[index],
                "probability_context_logprob_mean": context_means[index],
                "probability_context_token_count": context_counts[index],
                "probability_prior_logprob_sum": prior_sums[index],
                "probability_prior_logprob_mean": prior_means[index],
                "probability_prior_token_count": prior_counts[index],
            }
        )
    return decoded, scores


def main() -> None:
    parser = HfArgumentParser((InferenceArguments,))
    (args,) = parser.parse_args_into_dataclasses()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.max_length <= 0:
        raise ValueError("max_length must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.stop is not None and args.stop <= 0:
        raise ValueError("stop must be positive when provided")
    if args.steering_hook_scope not in STEERING_HOOK_SCOPES:
        raise ValueError(
            f"steering_hook_scope must be one of {sorted(STEERING_HOOK_SCOPES)}, got {args.steering_hook_scope!r}"
        )

    dataset_paths = [path.strip() for path in args.dataset_paths.split(",") if path.strip()]
    if not dataset_paths:
        raise ValueError("No dataset paths provided")
    missing_datasets = [path for path in dataset_paths if not Path(path).is_file()]
    if missing_datasets:
        raise FileNotFoundError("Evaluation datasets not found: " + ", ".join(missing_datasets))
    dataset_names = [Path(path).stem for path in dataset_paths]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("Dataset paths must have unique filename stems")
    if not Path(args.projection_path).is_dir():
        raise FileNotFoundError(f"Projection directory not found: {args.projection_path}")

    run = prepare_eval_run(
        category=args.eval_category,
        expid=args.expid,
        output_root=args.output_root,
        timestamp=args.timestamp,
        output_name=args.output_name,
    )
    configure_run_logging(run, kind="subspace-inference-v5")
    setup_seed(args.seed)
    device = get_device()
    runtime = describe_runtime(device)
    logger.info("Using device: %s", device)
    logger.info("Runtime: %s", runtime)
    logger.info("Evaluating datasets: %s", dataset_paths)
    model, processor = load_model_and_processor(
        args.model_name_or_path,
        lora_path=args.lora_name_or_path,
        merge_lora=True,
        torch_dtype=torch.bfloat16,
    )
    precision = describe_model_precision(model)
    unexpected_dtypes = set(precision["floating_parameter_counts_by_dtype"]) - {"bfloat16"}
    if unexpected_dtypes:
        raise RuntimeError(
            f"Inference requires all floating model parameters to use bfloat16, but found: {sorted(unexpected_dtypes)}"
        )
    model_revision = getattr(model.config, "_commit_hash", None)
    generation_policy = describe_eval_generation(model)
    logger.info("Inference precision: %s", precision)
    logger.info("Generation policy: %s", generation_policy)

    projection = LowRankOrthogonalProjection.from_pretrained(args.projection_path)
    projection.eval()
    steering_layers = [int(layer) for layer in args.steering_layer.split(",")]
    hooks: list[BinaryHook] = []
    for layer in steering_layers:
        hook = BinaryHook(
            projection,
            layer=layer,
            value_a=args.steering_prior_value,
            value_b=args.steering_context_value,
            prefill_only=True,
        )
        hook.attach(model)
        hooks.append(hook)
    logger.info(
        "Attached absolute DAS projection %s at layers %s (signed prior=%s, signed context=%s, generation scope=%s)",
        args.projection_path,
        steering_layers,
        args.steering_prior_value,
        args.steering_context_value,
        args.steering_hook_scope,
    )

    tasks: list[Task] = []
    for dataset_path in dataset_paths:
        task = Task(
            jsonl_path=dataset_path,
            image_root=args.image_root,
            processor=processor,
            max_length=args.max_length,
            stop=args.stop,
            training=False,
            use_image=args.use_image,
            use_instruction=args.use_instruction,
        )
        tasks.append(task)
        logger.info("[%s] Loaded %d samples", Path(dataset_path).stem, len(task))

    task_mixture = TaskMixture(tasks=tasks, shuffle=False)
    collator = CustomCollator(
        processor=processor,
        max_length=args.max_length,
        inference_mode=True,
    )
    dataloader = DataLoader(
        task_mixture,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    results: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(tqdm(dataloader, desc="Steering inference")):
        metadatas = batch.pop("_metadata")
        if batch_index == 0:
            logger.info(
                "Tokenized one sample %s",
                processor.tokenizer.decode(batch["input_ids"][0]),
            )
        context_answers = [metadata.context_answer.strip() for metadata in metadatas]
        prior_answers = [metadata.prior_answer.strip() for metadata in metadatas]
        modes = [metadata.mode for metadata in metadatas]
        response_prefixes = [metadata.response_prefix for metadata in metadatas]
        example_ids = [str(metadata.example_id or "") for metadata in metadatas]
        if any(not example_id for example_id in example_ids):
            raise ValueError("Every evaluation sample must have a stable example_id")

        mode_mask = torch.tensor([mode == "context" for mode in modes])
        for hook in hooks:
            hook.set_binary(mode_mask)
        with torch.inference_mode():
            predictions, scores = score_batch(
                model,
                batch,
                processor,
                context_answers,
                prior_answers,
                response_prefixes,
                hooks,
                device=device,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )

        for index, (prediction, score) in enumerate(zip(predictions, scores)):
            mode = modes[index]
            expected = context_answers[index] if mode == "context" else prior_answers[index]
            probability_context = float(score["probability_context_logprob_mean"])
            probability_prior = float(score["probability_prior_logprob_mean"])
            probability_margin = (
                probability_context - probability_prior if mode == "context" else probability_prior - probability_context
            )
            results.append(
                {
                    "example_id": example_ids[index],
                    "dataset": metadatas[index].dataset_name,
                    "image": metadatas[index].image_path,
                    "statement": metadatas[index].statement,
                    "question": metadatas[index].question,
                    "mode": mode,
                    "steering_value": (args.steering_context_value if mode == "context" else args.steering_prior_value),
                    "response_prefix": response_prefixes[index],
                    "context_answer": context_answers[index],
                    "prior_answer": prior_answers[index],
                    "expected": expected,
                    "prediction": prediction,
                    "expected_normalized": normalize_answer(expected),
                    "prediction_normalized": normalize_answer(prediction),
                    "exact_match_correct": exact_match(prediction, expected),
                    "probability_full_answer_correct": probability_margin > 0.0,
                    "probability_full_answer_margin": probability_margin,
                    "probability_context_logprob_sum": float(score["probability_context_logprob_sum"]),
                    "probability_context_logprob_mean": probability_context,
                    "probability_context_token_count": int(score["probability_context_token_count"]),
                    "probability_prior_logprob_sum": float(score["probability_prior_logprob_sum"]),
                    "probability_prior_logprob_mean": probability_prior,
                    "probability_prior_token_count": int(score["probability_prior_token_count"]),
                }
            )

    for hook in hooks:
        hook.remove()

    results_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        results_by_dataset[row["dataset"]].append(row)
    metrics_by_dataset = {dataset: compute_metrics(rows) for dataset, rows in results_by_dataset.items()}
    aggregate_metrics = compute_metrics(results)
    macro_metrics = compute_macro_metrics(metrics_by_dataset)

    output_data = {
        "config": {
            "model_name_or_path": args.model_name_or_path,
            "model_revision": model_revision,
            "lora_name_or_path": args.lora_name_or_path,
            "lora_merged": args.lora_name_or_path is not None,
            "projection_path": args.projection_path,
            "steering_layers": steering_layers,
            "intervention": "absolute_projection_replacement",
            "value_convention": "signed_hook_value",
            "steering_hook_scope": args.steering_hook_scope,
            "steering_prior_value": args.steering_prior_value,
            "steering_context_value": args.steering_context_value,
            "dataset_paths": dataset_paths,
            "image_root": args.image_root,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "stop": args.stop,
            "use_image": args.use_image,
            "use_instruction": args.use_instruction,
            "seed": args.seed,
            "dataset_order": "jsonl_source_order",
            "precision": precision,
            "runtime": runtime,
            "generation": generation_policy,
            "probability_reduction": "mean",
        },
        "metric_protocol": build_steering_metric_protocol(args.steering_hook_scope),
        "metrics": {
            "aggregate": aggregate_metrics,
            "macro_average": macro_metrics,
            "per_dataset": metrics_by_dataset,
        },
        "results": results,
    }
    output_path = write_json_result(
        run,
        output_data,
        kind="subspace-inference-v5",
    )
    logger.info("Results saved to: %s", output_path)

    if len(dataset_paths) > 1:
        for dataset_path in dataset_paths:
            dataset_name = Path(dataset_path).stem
            log_metrics(
                metrics_by_dataset[dataset_name],
                title=f"Dataset: {dataset_name}",
            )
        log_metrics(macro_metrics, title="Macro-average (Unweighted)")
        log_metrics(aggregate_metrics, title="Aggregate (Micro, All Samples)")
    else:
        log_metrics(
            aggregate_metrics,
            title=f"Dataset: {Path(dataset_paths[0]).stem}",
        )


if __name__ == "__main__":
    main()
