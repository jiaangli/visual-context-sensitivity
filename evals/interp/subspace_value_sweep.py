from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor, HfArgumentParser

from evals.interp.subspace_inference import (
    STEERING_HOOK_SCOPES,
    build_steering_metric_protocol,
    describe_model_precision,
    describe_runtime,
    exact_match,
    normalize_answer,
    score_batch,
)
from evals.sft.inference import compute_macro_metrics, compute_metrics, log_metrics
from src.eval_generation import describe_eval_generation
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

load_dotenv()
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def build_value_sweep_protocol(steering_hook_scope: str) -> dict[str, Any]:
    return {
        **build_steering_metric_protocol(steering_hook_scope),
        "version": "steering-value-sweep-v5",
        "selection": {
            "data": "one or more configurable validation datasets; no test result is consumed",
            "pairing": "context/prior mode rows joined by stable example_id",
            "multi_dataset_selection": "unweighted macro average across validation datasets",
            "recommended": (
                "maximize strict full-answer Pair EM among candidates that strictly "
                "beat the supplied base reference on both Pair EM and probability "
                "Pair Accuracy; break ties by probability Pair Accuracy and then "
                "smaller coefficient norm"
            ),
            "recommended_sft_closest": (
                "among candidates that strictly beat the supplied base reference on "
                "both pair metrics, minimize Euclidean distance to the supplied SFT "
                "reference after normalizing each axis by its Base-to-SFT gap"
            ),
        },
    }


VALUE_SWEEP_PROTOCOL = build_value_sweep_protocol("response_prefix")


@dataclass
class SweepArguments:
    expid: str = field(default="das-value-sweep")
    eval_category: str = field(default="steering")
    model_name_or_path: str = field(default="Qwen/Qwen3.5-4B")
    lora_name_or_path: str | None = field(default=None)
    dataset_path: str = field(
        default="dataset/whatifvis/spatial_temporal_val.jsonl",
        metadata={"help": "Comma-separated validation JSONL paths"},
    )
    image_root: str | None = field(default="dataset/whatifvis")
    projection_path: str = field(default="")
    steering_layer: str = field(
        default="0",
        metadata={"help": "Comma-separated 0-based layer indices"},
    )
    context_values: str = field(default="-10,-5,0,5,10")
    prior_values: str = field(default="-10,-5,0,5,10")
    steering_hook_scope: str = field(
        default="response_prefix",
        metadata={"help": ("response_prefix applies DAS only at the final response token...")},
    )
    joint_conditions: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional comma-separated context:prior pairs. When supplied, "
                "evaluate each pair directly in interleaved dataset order."
            )
        },
    )
    base_reference_path: str | None = field(default=None)
    sft_reference_path: str | None = field(default=None)
    output_root: str = field(default=DEFAULT_EVAL_ROOT)
    timestamp: str | None = field(default=None)
    output_name: str = field(default="value_sweep.json")
    batch_size: int = field(default=16)
    max_length: int = field(default=512)
    max_new_tokens: int = field(default=16)
    max_rows_per_dataset: int | None = field(
        default=None,
        metadata={
            "help": (
                "Optional validation-search limit in raw JSONL rows per dataset. "
                "Each raw row still contributes both context and prior modes."
            )
        },
    )
    resume_checkpoint_path: str | None = field(
        default=None,
        metadata={"help": "Optional compatible per-axis checkpoint to resume"},
    )
    use_image: bool = field(default=True)
    use_instruction: bool = field(
        default=False,
        metadata={"help": "Neutral prompts are required for mode steering."},
    )
    seed: int = field(default=42)


def parse_values(specification: str) -> list[float]:
    """Parse a comma-separated, finite, order-preserving value list."""
    values: list[float] = []
    seen: set[float] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not math.isfinite(value):
            raise ValueError(f"Steering values must be finite, got {item!r}")
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError("At least one steering value is required")
    return values


def parse_dataset_paths(specification: str) -> list[str]:
    """Parse unique comma-separated dataset paths while preserving order."""
    paths: list[str] = []
    seen: set[str] = set()
    for item in specification.split(","):
        path = item.strip()
        if path and path not in seen:
            paths.append(path)
            seen.add(path)
    if not paths:
        raise ValueError("At least one validation dataset path is required")
    return paths


def parse_joint_conditions(specification: str) -> list[tuple[float, float]]:
    """Parse unique ``context:prior`` pairs while preserving their order."""
    conditions: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError("Joint conditions must use comma-separated context:prior pairs")
        condition = (float(parts[0]), float(parts[1]))
        if not all(math.isfinite(value) for value in condition):
            raise ValueError(f"Steering values must be finite, got {item!r}")
        if condition not in seen:
            conditions.append(condition)
            seen.add(condition)
    if not conditions:
        raise ValueError("At least one joint steering condition is required")
    return conditions


def parse_steering_layers(specification: str) -> list[int]:
    """Parse unique comma-separated layer indices while preserving order."""
    layers: list[int] = []
    seen: set[int] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        layer = int(item)
        if layer < 0:
            raise ValueError(f"Steering layers must be non-negative, got {layer}")
        if layer not in seen:
            layers.append(layer)
            seen.add(layer)
    if not layers:
        raise ValueError("At least one steering layer is required")
    return layers


def _reference_metrics(
    path: str | None,
    *,
    expected_dataset_paths: Sequence[str],
    expected_lora: bool,
    expected_model_name_or_path: str | None = None,
    expected_use_image: bool | None = None,
    expected_seed: int | None = None,
    max_rows_per_dataset: int | None = None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    reference_path = Path(path)
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference result not found: {reference_path}")
    with reference_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("config", {})
    dataset_paths = config.get("dataset_paths")
    if dataset_paths != list(expected_dataset_paths):
        raise ValueError(
            f"Reference {reference_path} used datasets {dataset_paths}, expected {list(expected_dataset_paths)}"
        )
    has_lora = config.get("lora_name_or_path") is not None
    if has_lora != expected_lora:
        expected = "SFT" if expected_lora else "base"
        raise ValueError(f"Reference {reference_path} is not a {expected} run")
    if config.get("use_instruction") is not True:
        raise ValueError(f"Reference {reference_path} must use explicit instructions")
    if expected_model_name_or_path is not None and config.get("model_name_or_path") != expected_model_name_or_path:
        raise ValueError(
            f"Reference {reference_path} used model "
            f"{config.get('model_name_or_path')!r}, expected "
            f"{expected_model_name_or_path!r}"
        )
    if expected_use_image is not None and config.get("use_image") is not expected_use_image:
        raise ValueError(
            f"Reference {reference_path} has use_image={config.get('use_image')!r}, expected {expected_use_image!r}"
        )
    if expected_seed is not None and int(config.get("seed", -1)) != expected_seed:
        raise ValueError(f"Reference {reference_path} used seed {config.get('seed')!r}, expected {expected_seed}")
    metric_scope = "macro_average" if len(expected_dataset_paths) > 1 else "aggregate"
    if max_rows_per_dataset is None:
        metrics = payload["metrics"][metric_scope]
    else:
        rows = limit_rows_per_dataset(
            payload.get("results", []),
            max_rows_per_dataset=max_rows_per_dataset,
        )
        if not rows:
            raise ValueError(f"Reference {reference_path} has no rows available for truncation")
        metrics = summarize_condition_rows(rows)["metrics"]
    return {
        "path": str(reference_path),
        "config": payload["config"],
        "metric_protocol": payload["metric_protocol"],
        "metric_scope": metric_scope,
        "max_rows_per_dataset": max_rows_per_dataset,
        "metrics": metrics,
    }


def limit_rows_per_dataset(
    rows: Sequence[dict[str, Any]],
    *,
    max_rows_per_dataset: int,
) -> list[dict[str, Any]]:
    """Keep both mode rows for the first N raw examples in each dataset."""
    if max_rows_per_dataset <= 0:
        raise ValueError("max_rows_per_dataset must be positive")
    selected_ids: dict[str, list[str]] = defaultdict(list)
    selected_sets: dict[str, set[str]] = defaultdict(set)
    selected_rows: list[dict[str, Any]] = []
    for row in rows:
        dataset = str(row["dataset"])
        example_id = str(row["example_id"])
        if example_id not in selected_sets[dataset]:
            if len(selected_ids[dataset]) >= max_rows_per_dataset:
                continue
            selected_ids[dataset].append(example_id)
            selected_sets[dataset].add(example_id)
        if example_id in selected_sets[dataset]:
            selected_rows.append(row)
    return selected_rows


def write_mode_checkpoint(
    path: Path,
    *,
    config: Mapping[str, Any],
    mode_evaluations: Sequence[dict[str, Any]],
) -> None:
    """Atomically persist completed axis evaluations for interruption-safe sweeps."""
    payload = {
        "schema_version": 1,
        "config": dict(config),
        "mode_evaluations": list(mode_evaluations),
    }
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def load_mode_checkpoint(
    path: str | Path,
    *,
    expected_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load completed axis evaluations after validating sweep compatibility."""
    checkpoint_path = Path(path)
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported checkpoint schema: {checkpoint_path}")
    actual_config = payload.get("config")
    if actual_config != dict(expected_config):
        raise ValueError(
            f"Checkpoint configuration mismatch: {checkpoint_path}; "
            f"expected {dict(expected_config)!r}, got {actual_config!r}"
        )
    evaluations = payload.get("mode_evaluations")
    if not isinstance(evaluations, list):
        raise TypeError(f"Invalid mode_evaluations in checkpoint: {checkpoint_path}")
    return evaluations


def summarize_condition_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate, per-dataset, and macro metrics for one condition."""
    rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_dataset[str(row["dataset"])].append(row)
    per_dataset = {
        dataset_name: compute_metrics(dataset_rows) for dataset_name, dataset_rows in rows_by_dataset.items()
    }
    aggregate = compute_metrics(rows)
    macro_average = compute_macro_metrics(per_dataset)
    selection_metrics = macro_average if len(per_dataset) > 1 else aggregate
    return {
        "metrics": selection_metrics,
        "aggregate_metrics": aggregate,
        "macro_average_metrics": macro_average,
        "per_dataset_metrics": per_dataset,
    }


def _metric(record: Mapping[str, Any], metric: str, key: str) -> float:
    return float(record["metrics"][metric][key])


def select_sweep_conditions(
    sweep: Sequence[dict[str, Any]],
    base_metrics: Mapping[str, Mapping[str, float | int]] | None = None,
    sft_metrics: Mapping[str, Mapping[str, float | int]] | None = None,
) -> dict[str, Any]:
    """Select unconstrained and baseline-constrained recommendations."""
    if not sweep:
        raise ValueError("Cannot select from an empty sweep")

    def coefficient_tiebreak(record: Mapping[str, Any]) -> float:
        condition = record["condition"]
        return -(abs(float(condition["context_value"])) + abs(float(condition["prior_value"])))

    best_exact = max(
        sweep,
        key=lambda record: (
            _metric(record, "exact_match", "pair_accuracy"),
            _metric(record, "probability_full_answer", "pair_accuracy"),
            _metric(record, "exact_match", "overall_accuracy"),
            coefficient_tiebreak(record),
        ),
    )
    best_probability = max(
        sweep,
        key=lambda record: (
            _metric(record, "probability_full_answer", "pair_accuracy"),
            _metric(record, "exact_match", "pair_accuracy"),
            _metric(record, "probability_full_answer", "overall_accuracy"),
            coefficient_tiebreak(record),
        ),
    )

    eligible = list(sweep)
    constraint = None
    if base_metrics is not None:
        base_exact = float(base_metrics["exact_match"]["pair_accuracy"])
        base_probability = float(base_metrics["probability_full_answer"]["pair_accuracy"])
        constraint = {
            "strictly_greater_than_base_exact_pair": base_exact,
            "strictly_greater_than_base_probability_pair": base_probability,
        }
        eligible = [
            record
            for record in sweep
            if _metric(record, "exact_match", "pair_accuracy") > base_exact
            and _metric(
                record,
                "probability_full_answer",
                "pair_accuracy",
            )
            > base_probability
        ]

    recommended = None
    if eligible:
        recommended = max(
            eligible,
            key=lambda record: (
                _metric(record, "exact_match", "pair_accuracy"),
                _metric(record, "probability_full_answer", "pair_accuracy"),
                _metric(record, "exact_match", "overall_accuracy"),
                coefficient_tiebreak(record),
            ),
        )

    recommended_sft_closest = None
    recommended_sft_closest_distance = None
    sft_closeness = None
    if sft_metrics is not None:
        if base_metrics is None:
            raise ValueError("SFT-closeness selection requires base metrics")
        base_exact = float(base_metrics["exact_match"]["pair_accuracy"])
        base_probability = float(base_metrics["probability_full_answer"]["pair_accuracy"])
        sft_exact = float(sft_metrics["exact_match"]["pair_accuracy"])
        sft_probability = float(sft_metrics["probability_full_answer"]["pair_accuracy"])

        exact_scale = max(abs(sft_exact - base_exact), 1e-12)
        probability_scale = max(abs(sft_probability - base_probability), 1e-12)

        def sft_distance(record: Mapping[str, Any]) -> float:
            exact_delta = (_metric(record, "exact_match", "pair_accuracy") - sft_exact) / exact_scale
            probability_delta = (
                _metric(
                    record,
                    "probability_full_answer",
                    "pair_accuracy",
                )
                - sft_probability
            ) / probability_scale
            return math.hypot(exact_delta, probability_delta)

        sft_closeness = {
            "target_exact_pair": sft_exact,
            "target_probability_pair": sft_probability,
            "normalization_exact_base_to_sft_gap": exact_scale,
            "normalization_probability_base_to_sft_gap": probability_scale,
            "distance": "normalized_euclidean_pair_metric_distance",
        }
        if eligible:
            recommended_sft_closest = min(
                eligible,
                key=lambda record: (
                    sft_distance(record),
                    -_metric(record, "exact_match", "pair_accuracy"),
                    -_metric(
                        record,
                        "probability_full_answer",
                        "pair_accuracy",
                    ),
                    -_metric(record, "exact_match", "overall_accuracy"),
                    -coefficient_tiebreak(record),
                ),
            )
            recommended_sft_closest_distance = sft_distance(recommended_sft_closest)

    return {
        "constraint": constraint,
        "eligible_count": len(eligible),
        "recommended_exact_first": recommended,
        "recommended_sft_closest": recommended_sft_closest,
        "recommended_sft_closest_distance": recommended_sft_closest_distance,
        "sft_closeness": sft_closeness,
        "best_exact_unconstrained": best_exact,
        "best_probability_unconstrained": best_probability,
    }


def _cache_batches(
    task: Task,
    processor: AutoProcessor,
    *,
    indices: Sequence[int],
    batch_size: int,
    max_length: int,
) -> list[tuple[dict[str, torch.Tensor], list[Any]]]:
    loader = DataLoader(
        Subset(task, list(indices)),
        batch_size=batch_size,
        collate_fn=CustomCollator(
            processor=processor,
            max_length=max_length,
            inference_mode=True,
        ),
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    cached: list[tuple[dict[str, torch.Tensor], list[Any]]] = []
    for batch in loader:
        metadatas = batch.pop("_metadata")
        cached.append((batch, metadatas))
    return cached


def _evaluate_condition(
    *,
    model,
    processor: AutoProcessor,
    cached_batches: Sequence[tuple[dict[str, torch.Tensor], list[Any]]],
    hooks: Sequence[BinaryHook],
    context_value: float,
    prior_value: float,
    mode_override: str | None,
    device: str,
    max_length: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    if mode_override not in {None, "context", "prior"}:
        raise ValueError(f"Unknown mode: {mode_override}")
    for hook in hooks:
        hook.prior_value = prior_value
        hook.context_value = context_value

    rows: list[dict[str, Any]] = []
    for inputs, metadatas in cached_batches:
        modes = [mode_override if mode_override is not None else metadata.mode for metadata in metadatas]
        mode_mask = torch.tensor([mode == "context" for mode in modes])
        for hook in hooks:
            hook.set_binary(mode_mask)

        context_answers = [metadata.context_answer.strip() for metadata in metadatas]
        prior_answers = [metadata.prior_answer.strip() for metadata in metadatas]
        response_prefixes = [metadata.response_prefix for metadata in metadatas]
        example_ids = [str(metadata.example_id or "") for metadata in metadatas]
        if any(not example_id for example_id in example_ids):
            raise ValueError("Every validation sample must have a stable example_id")

        with torch.inference_mode():
            predictions, scores = score_batch(
                model,
                inputs,
                processor,
                context_answers,
                prior_answers,
                response_prefixes,
                hooks,
                device=device,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
            )

        for metadata, mode, prediction, score in zip(
            metadatas,
            modes,
            predictions,
            scores,
        ):
            context_answer = metadata.context_answer.strip()
            prior_answer = metadata.prior_answer.strip()
            expected = context_answer if mode == "context" else prior_answer
            probability_context = float(score["probability_context_logprob_mean"])
            probability_prior = float(score["probability_prior_logprob_mean"])
            probability_margin = (
                probability_context - probability_prior
                if mode == "context"
                else probability_prior - probability_context
            )
            rows.append(
                {
                    "example_id": str(metadata.example_id),
                    "dataset": metadata.dataset_name,
                    "image": metadata.image_path,
                    "statement": metadata.statement,
                    "question": metadata.question,
                    "mode": mode,
                    "steering_value": (context_value if mode == "context" else prior_value),
                    "response_prefix": metadata.response_prefix,
                    "context_answer": context_answer,
                    "prior_answer": prior_answer,
                    "expected": expected,
                    "prediction": prediction,
                    "expected_normalized": normalize_answer(expected),
                    "prediction_normalized": normalize_answer(prediction),
                    "exact_match_correct": exact_match(
                        prediction,
                        expected,
                    ),
                    "probability_full_answer_correct": probability_margin > 0.0,
                    "probability_full_answer_margin": probability_margin,
                    "probability_context_logprob_mean": probability_context,
                    "probability_prior_logprob_mean": probability_prior,
                    "probability_context_token_count": int(score["probability_context_token_count"]),
                    "probability_prior_token_count": int(score["probability_prior_token_count"]),
                }
            )
    return rows


def _evaluate_mode_value(
    *,
    model,
    processor: AutoProcessor,
    cached_batches: Sequence[tuple[dict[str, torch.Tensor], list[Any]]],
    hooks: Sequence[BinaryHook],
    mode: str,
    value: float,
    device: str,
    max_length: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    return _evaluate_condition(
        model=model,
        processor=processor,
        cached_batches=cached_batches,
        hooks=hooks,
        context_value=value,
        prior_value=value,
        mode_override=mode,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )


def main() -> None:
    parser = HfArgumentParser((SweepArguments,))
    (args,) = parser.parse_args_into_dataclasses()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.max_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("length limits must be positive")
    if args.max_rows_per_dataset is not None and args.max_rows_per_dataset <= 0:
        raise ValueError("max_rows_per_dataset must be positive when provided")
    if args.use_instruction:
        raise ValueError("Value sweeps require a neutral prompt without mode instructions")
    if args.steering_hook_scope not in STEERING_HOOK_SCOPES:
        raise ValueError(
            f"steering_hook_scope must be one of {sorted(STEERING_HOOK_SCOPES)}, got {args.steering_hook_scope!r}"
        )
    dataset_paths = parse_dataset_paths(args.dataset_path)
    missing_datasets = [path for path in dataset_paths if not Path(path).is_file()]
    if missing_datasets:
        raise FileNotFoundError("Validation datasets not found: " + ", ".join(missing_datasets))
    dataset_names = [Path(path).stem for path in dataset_paths]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("Dataset paths must have unique filename stems")
    if not Path(args.projection_path).is_dir():
        raise FileNotFoundError(f"Projection not found: {args.projection_path}")

    context_values = parse_values(args.context_values)
    prior_values = parse_values(args.prior_values)
    steering_layers = parse_steering_layers(args.steering_layer)
    joint_conditions = parse_joint_conditions(args.joint_conditions) if args.joint_conditions is not None else None

    base_reference = _reference_metrics(
        args.base_reference_path,
        expected_dataset_paths=dataset_paths,
        expected_lora=False,
        expected_model_name_or_path=args.model_name_or_path,
        expected_use_image=args.use_image,
        expected_seed=args.seed,
        max_rows_per_dataset=args.max_rows_per_dataset,
    )
    sft_reference = _reference_metrics(
        args.sft_reference_path,
        expected_dataset_paths=dataset_paths,
        expected_lora=True,
        expected_model_name_or_path=args.model_name_or_path,
        expected_use_image=args.use_image,
        expected_seed=args.seed,
        max_rows_per_dataset=args.max_rows_per_dataset,
    )

    run = prepare_eval_run(
        category=args.eval_category,
        expid=args.expid,
        output_root=args.output_root,
        timestamp=args.timestamp,
        output_name=args.output_name,
    )
    configure_run_logging(run, kind="subspace-value-sweep-v5")
    setup_seed(args.seed)
    device = get_device()
    runtime = describe_runtime(device)
    logger.info(
        "Loading application model %s on %s (LoRA=%s)",
        args.model_name_or_path,
        device,
        args.lora_name_or_path,
    )
    logger.info("Validation-only datasets: %s", dataset_paths)
    if args.max_rows_per_dataset is not None:
        logger.info(
            "Validation search uses the first %d raw rows per dataset",
            args.max_rows_per_dataset,
        )

    if joint_conditions is None:
        logger.info("Context values: %s", context_values)
        logger.info("Prior values: %s", prior_values)
    else:
        logger.info("Direct joint conditions: %s", joint_conditions)

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
            f"Value sweep requires all floating parameters in bfloat16, found {sorted(unexpected_dtypes)}"
        )

    tasks = [
        Task(
            jsonl_path=dataset_path,
            image_root=args.image_root,
            processor=processor,
            max_length=args.max_length,
            stop=(None if args.max_rows_per_dataset is None else 2 * args.max_rows_per_dataset),
            training=False,
            use_image=args.use_image,
            use_instruction=False,
        )
        for dataset_path in dataset_paths
    ]
    task = tasks[0] if len(tasks) == 1 else TaskMixture(tasks=tasks, shuffle=False)
    if joint_conditions is None:
        # With no mode instruction, context and prior prompts are byte-identical.
        # Cache only the even/context copy and reuse it for both mode sweeps.
        cache_indices = range(0, len(task), 2)
        cached_examples = len(task) // 2
    else:
        cache_indices = range(len(task))
        cached_examples = len(task)
    cached_batches = _cache_batches(
        task,
        processor,
        indices=cache_indices,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    logger.info(
        "Cached %d unique validation examples in %d batches",
        cached_examples,
        len(cached_batches),
    )

    projection = LowRankOrthogonalProjection.from_pretrained(args.projection_path)
    projection.eval()
    hooks: list[BinaryHook] = []
    for layer in steering_layers:
        hook = BinaryHook(
            projection,
            layer=layer,
            value_a=0.0,
            value_b=0.0,
            prefill_only=True,
        )
        hook.attach(model)
        hooks.append(hook)

    checkpoint_config = {
        "model_name_or_path": args.model_name_or_path,
        "lora_name_or_path": args.lora_name_or_path,
        "projection_path": str(Path(args.projection_path).resolve()),
        "steering_layers": steering_layers,
        "steering_hook_scope": args.steering_hook_scope,
        "dataset_paths": dataset_paths,
        "max_rows_per_dataset": args.max_rows_per_dataset,
        "context_values": context_values,
        "prior_values": prior_values,
        "seed": args.seed,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "use_image": args.use_image,
    }
    checkpoint_path = run.artifact_path(f"{run.result_path.stem}.mode_checkpoint.json")
    mode_evaluations: list[dict[str, Any]] = []
    sweep: list[dict[str, Any]] = []
    try:
        if joint_conditions is not None:
            if args.resume_checkpoint_path is not None:
                raise ValueError("resume_checkpoint_path is supported only for separable axis sweeps")
            for condition_index, (context_value, prior_value) in enumerate(
                joint_conditions,
                start=1,
            ):
                logger.info(
                    "Evaluating joint context=%g prior=%g (%d/%d)",
                    context_value,
                    prior_value,
                    condition_index,
                    len(joint_conditions),
                )
                rows = _evaluate_condition(
                    model=model,
                    processor=processor,
                    cached_batches=cached_batches,
                    hooks=hooks,
                    context_value=context_value,
                    prior_value=prior_value,
                    mode_override=None,
                    device=device,
                    max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens,
                )
                sweep.append(
                    {
                        "condition": {
                            "context_value": context_value,
                            "prior_value": prior_value,
                        },
                        **summarize_condition_rows(rows),
                        "results": rows,
                    }
                )
        else:
            mode_rows: dict[tuple[str, float], list[dict[str, Any]]] = {}
            if args.resume_checkpoint_path is not None:
                mode_evaluations = load_mode_checkpoint(
                    args.resume_checkpoint_path,
                    expected_config=checkpoint_config,
                )
                expected_keys = {
                    *(("context", value) for value in context_values),
                    *(("prior", value) for value in prior_values),
                }
                for evaluation in mode_evaluations:
                    condition = evaluation.get("condition", {})
                    key = (str(condition.get("mode")), float(condition.get("value")))
                    if key not in expected_keys:
                        raise ValueError(f"Unexpected condition {key!r} in resume checkpoint")
                    if key in mode_rows:
                        raise ValueError(f"Duplicate condition {key!r} in resume checkpoint")
                    rows = evaluation.get("results")
                    if not isinstance(rows, list):
                        raise TypeError(f"Missing result rows for {key!r} in resume checkpoint")
                    mode_rows[key] = rows
                logger.info(
                    "Resumed %d completed axis evaluations from %s",
                    len(mode_rows),
                    args.resume_checkpoint_path,
                )

            for mode, values in (
                ("context", context_values),
                ("prior", prior_values),
            ):
                for value_index, value in enumerate(values, start=1):
                    key = (mode, value)
                    if key in mode_rows:
                        logger.info(
                            "Skipping completed %s value %g (%d/%d)",
                            mode,
                            value,
                            value_index,
                            len(values),
                        )
                        continue
                    logger.info(
                        "Evaluating %s value %g (%d/%d)",
                        mode,
                        value,
                        value_index,
                        len(values),
                    )
                    rows = _evaluate_mode_value(
                        model=model,
                        processor=processor,
                        cached_batches=cached_batches,
                        hooks=hooks,
                        mode=mode,
                        value=value,
                        device=device,
                        max_length=args.max_length,
                        max_new_tokens=args.max_new_tokens,
                    )
                    mode_rows[key] = rows
                    mode_evaluations.append(
                        {
                            "condition": {
                                "mode": mode,
                                "value": value,
                            },
                            **summarize_condition_rows(rows),
                            "results": rows,
                        }
                    )
                    write_mode_checkpoint(
                        checkpoint_path,
                        config=checkpoint_config,
                        mode_evaluations=mode_evaluations,
                    )

            for context_value, prior_value in product(
                context_values,
                prior_values,
            ):
                rows = [
                    *mode_rows[("context", context_value)],
                    *mode_rows[("prior", prior_value)],
                ]
                sweep.append(
                    {
                        "condition": {
                            "context_value": context_value,
                            "prior_value": prior_value,
                        },
                        **summarize_condition_rows(rows),
                    }
                )
    finally:
        for hook in hooks:
            hook.remove()

    selections = select_sweep_conditions(
        sweep,
        None if base_reference is None else base_reference["metrics"],
        None if sft_reference is None else sft_reference["metrics"],
    )
    recommended = selections["recommended_exact_first"]
    if recommended is not None:
        condition = recommended["condition"]
        logger.info(
            "Recommended validation values: context=%g prior=%g",
            condition["context_value"],
            condition["prior_value"],
        )
        log_metrics(recommended["metrics"], title="Recommended validation condition")
    else:
        logger.warning("No candidate strictly beat the supplied base reference on both pair metrics")

    recommended_sft_closest = selections["recommended_sft_closest"]
    if recommended_sft_closest is not None:
        condition = recommended_sft_closest["condition"]
        logger.info(
            "Closest-to-SFT eligible values: context=%g prior=%g distance=%g",
            condition["context_value"],
            condition["prior_value"],
            selections["recommended_sft_closest_distance"],
        )
        log_metrics(
            recommended_sft_closest["metrics"],
            title="Closest-to-SFT eligible validation condition",
        )

    output_data = {
        "config": {
            "model_name_or_path": args.model_name_or_path,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "lora_name_or_path": args.lora_name_or_path,
            "lora_merged": args.lora_name_or_path is not None,
            "projection_path": args.projection_path,
            "steering_layer": (steering_layers[0] if len(steering_layers) == 1 else args.steering_layer),
            "steering_layers": steering_layers,
            "intervention": "absolute_projection_replacement",
            "value_convention": "signed_hook_value",
            "steering_hook_scope": args.steering_hook_scope,
            "dataset_path": args.dataset_path,
            "dataset_paths": dataset_paths,
            "selection_metric_scope": ("macro_average" if len(dataset_paths) > 1 else "aggregate"),
            "selection_split": "validation",
            "image_root": args.image_root,
            "context_values": context_values,
            "prior_values": prior_values,
            "joint_conditions": joint_conditions,
            "use_image": args.use_image,
            "use_instruction": False,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "max_rows_per_dataset": args.max_rows_per_dataset,
            "resume_checkpoint_path": args.resume_checkpoint_path,
            "mode_checkpoint_path": (str(checkpoint_path) if joint_conditions is None else None),
            "seed": args.seed,
            "common_random_numbers_per_condition": True,
            "precision": precision,
            "runtime": runtime,
            "generation": describe_eval_generation(model),
        },
        "metric_protocol": build_value_sweep_protocol(args.steering_hook_scope),
        "references": {
            "base": base_reference,
            "sft": sft_reference,
        },
        "selections": selections,
        "sweep": sweep,
        "mode_evaluations": mode_evaluations,
    }
    output_path = write_json_result(
        run,
        output_data,
        kind="subspace-value-sweep-v5",
    )
    logger.info("Value sweep saved to %s", output_path)


if __name__ == "__main__":
    main()
