import logging
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import transformers
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, HfArgumentParser

from src.eval_generation import build_eval_generation_kwargs, describe_eval_generation
from src.eval_results import (
    DEFAULT_EVAL_ROOT,
    configure_run_logging,
    prepare_eval_run,
    write_json_result,
)
from src.model_utils import get_device, load_model_and_processor, setup_seed
from src.sft.data_process import CustomCollator, Task, TaskMixture
from src.sft.generation_eval import exact_match, normalize_answer
from src.sft.preference_eval import (
    build_probability_inputs,
    sequence_logprobs,
)
from src.sft.preference_eval import (
    candidate_token_ids as _candidate_token_ids,
)

load_dotenv()
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

METRIC_PROTOCOL = {
    "version": "sft-eval-v3",
    "exact_match": {
        "normalization": "NFKC, casefold, replace non-letter/digit runs with spaces",
        "comparison": "full-string equality",
    },
    "probability_full_answer": {
        "conditioning": "each canonical candidate's own preceding answer tokens",
        "tokens": "canonical answer fragment only; excludes punctuation/chat suffix",
        "reduction": "mean token log-probability",
    },
    "pair": "both context and prior prompts must be correct for the same example_id",
}


# Arguments
@dataclass
class InferenceArguments:
    expid: str = field(
        default="sft-eval",
        metadata={"help": "Experiment id used to group related eval jobs"},
    )
    eval_category: str = field(default="sft", metadata={"help": "Experiment result category"})
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Path to the fine-tuned model (or base model)"},
    )
    lora_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to LoRA adapter if not included in model_name_or_path"},
    )
    dataset_paths: str = field(
        default="dataset/whatifvis/spatial_temporal_test.jsonl",
        metadata={"help": "Comma-separated paths to evaluation JSONL files"},
    )
    image_root: Optional[str] = field(
        default="dataset/whatifvis",
        metadata={"help": "Optional root to prepend to image paths"},
    )
    output_root: str = field(
        default=DEFAULT_EVAL_ROOT,
        metadata={"help": "Root directory for eval artifacts"},
    )
    timestamp: Optional[str] = field(
        default=None,
        metadata={"help": "Shared UTC run timestamp (YYYYMMDDTHHMMSSZ); generated when omitted"},
    )
    output_name: str = field(
        default="results.json",
        metadata={"help": "Result filename inside the expid/timestamp directory"},
    )
    batch_size: int = field(default=4, metadata={"help": "Batch size for generation"})
    max_length: int = field(default=512, metadata={"help": "Maximum input sequence length"})
    max_new_tokens: int = field(
        default=16,
        metadata={"help": "Maximum generated tokens for exact-match evaluation"},
    )
    stop: Optional[int] = field(
        default=None, metadata={"help": "Optional limit on number of samples to evaluate (for debugging)"}
    )
    use_image: bool = field(default=True, metadata={"help": "Whether to use images"})
    use_instruction: bool = field(default=True, metadata={"help": "Whether to use instruction"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})


def _autocast(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def describe_model_precision(model) -> dict[str, Any]:
    """Describe the floating-point precision used by model parameters.

    Log-probability reductions are intentionally performed in float32 after the
    model forward pass; this does not change the model's inference precision.
    """

    parameter_counts: dict[str, int] = defaultdict(int)
    for parameter in model.parameters():
        if parameter.is_floating_point():
            dtype_name = str(parameter.dtype).removeprefix("torch.")
            parameter_counts[dtype_name] += parameter.numel()

    def module_weight_dtype(module) -> str | None:
        weight = getattr(module, "weight", None)
        if weight is None:
            return None
        return str(weight.dtype).removeprefix("torch.")

    return {
        "requested_model_dtype": "bfloat16",
        "autocast_dtype": "bfloat16",
        "floating_parameter_counts_by_dtype": dict(sorted(parameter_counts.items())),
        "input_embedding_dtype": module_weight_dtype(model.get_input_embeddings()),
        "output_embedding_dtype": module_weight_dtype(model.get_output_embeddings()),
        "score_reduction_dtype": "float32",
    }


def describe_runtime(device: str) -> dict[str, Any]:
    """Record the software and accelerator details that affect reproducibility."""

    runtime: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "device": device,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if device == "cuda":
        device_index = torch.cuda.current_device()
        runtime.update(
            {
                "cuda_device_index": device_index,
                "cuda_device_name": torch.cuda.get_device_name(device_index),
                "cuda_device_capability": list(torch.cuda.get_device_capability(device_index)),
                "cuda_bfloat16_supported": torch.cuda.is_bf16_supported(),
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return runtime


def _probability_scores(
    model,
    inputs: Mapping[str, torch.Tensor],
    candidate_token_ids: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    max_length: int,
    device: str,
) -> Tuple[List[float], List[float], List[int]]:
    candidate_inputs, labels = build_probability_inputs(
        inputs,
        candidate_token_ids,
        pad_token_id=pad_token_id,
        max_length=max_length,
    )
    with _autocast(device):
        output = model(
            **candidate_inputs,
            use_cache=False,
            return_dict=True,
        )

    sums, means, counts = sequence_logprobs(output.logits, labels)
    return (
        sums.detach().cpu().tolist(),
        means.detach().cpu().tolist(),
        counts.detach().cpu().tolist(),
    )


def score_batch(
    model,
    inputs: Dict[str, torch.Tensor],
    processor: AutoProcessor,
    ctx_answers: List[str],
    prior_answers: List[str],
    response_prefixes: List[str],
    device: str,
    max_length: int,
    max_new_tokens: int,
) -> Tuple[List[str], List[Dict[str, float | int]]]:
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    if "gemma" in processor.tokenizer.name_or_path.lower():
        inputs["token_type_ids"] = inputs["mm_token_type_ids"]

    input_ids = inputs["input_ids"]
    context_token_ids = _candidate_token_ids(
        processor,
        ctx_answers,
        response_prefixes,
    )
    prior_token_ids = _candidate_token_ids(
        processor,
        prior_answers,
        response_prefixes,
    )

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
        context_token_ids,
        pad_token_id=pad_token_id,
        max_length=max_length,
        device=device,
    )
    prior_sums, prior_means, prior_counts = _probability_scores(
        model,
        inputs,
        prior_token_ids,
        pad_token_id=pad_token_id,
        max_length=max_length,
        device=device,
    )

    scores: List[Dict[str, float | int]] = []
    for index in range(len(ctx_answers)):
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


# Metrics computation


def _compute_binary_metric(
    results: Sequence[Mapping[str, Any]],
    value_key: str,
) -> Dict[str, float | int]:
    context_rows = [row for row in results if row["mode"] == "context"]
    prior_rows = [row for row in results if row["mode"] == "prior"]

    def average(rows: Sequence[Mapping[str, Any]]) -> float:
        return sum(float(row[value_key]) for row in rows) / len(rows) if rows else 0.0

    paired: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in results:
        mode = str(row["mode"])
        if mode not in {"context", "prior"}:
            raise ValueError(f"Unknown evaluation mode: {mode!r}")
        example_id = str(row.get("example_id") or "")
        if not example_id:
            raise ValueError("Every result row must contain a non-empty example_id")
        pair = paired.setdefault(example_id, {})
        if mode in pair:
            raise ValueError(f"Duplicate {mode} result for example_id={example_id!r}")
        pair[mode] = row

    complete_pairs = [pair for pair in paired.values() if {"context", "prior"} <= pair.keys()]
    pair_accuracy = (
        sum(bool(pair["context"][value_key]) and bool(pair["prior"][value_key]) for pair in complete_pairs)
        / len(complete_pairs)
        if complete_pairs
        else 0.0
    )
    return {
        "context_accuracy": average(context_rows),
        "prior_accuracy": average(prior_rows),
        "overall_accuracy": average(results),
        "pair_accuracy": pair_accuracy,
        "n_samples": len(results),
        "n_pairs": len(complete_pairs),
        "n_incomplete_pairs": len(paired) - len(complete_pairs),
    }


METRIC_VALUE_KEYS = {
    "exact_match": "exact_match_correct",
    "probability_full_answer": "probability_full_answer_correct",
}


def compute_metrics(results: List[dict]) -> Dict[str, Dict[str, float | int]]:
    """Compute each versioned metric and pair rows by stable example ID."""

    return {
        metric_name: _compute_binary_metric(results, value_key) for metric_name, value_key in METRIC_VALUE_KEYS.items()
    }


def compute_macro_metrics(
    metrics_by_dataset: Mapping[str, Mapping[str, Mapping[str, float | int]]],
) -> Dict[str, Dict[str, float | int]]:
    if not metrics_by_dataset:
        return compute_metrics([])

    macro: Dict[str, Dict[str, float | int]] = {}
    accuracy_keys = (
        "context_accuracy",
        "prior_accuracy",
        "overall_accuracy",
        "pair_accuracy",
    )
    count_keys = ("n_samples", "n_pairs", "n_incomplete_pairs")
    for metric_name in METRIC_VALUE_KEYS:
        macro[metric_name] = {
            key: sum(float(dataset_metrics[metric_name][key]) for dataset_metrics in metrics_by_dataset.values())
            / len(metrics_by_dataset)
            for key in accuracy_keys
        }
        macro[metric_name].update(
            {
                key: sum(int(dataset_metrics[metric_name][key]) for dataset_metrics in metrics_by_dataset.values())
                for key in count_keys
            }
        )
    return macro


def log_metrics(
    metrics: Mapping[str, Mapping[str, float | int]],
    title: str = "Results",
) -> None:
    """Pretty-print all explicitly named evaluation metrics."""

    labels = {
        "exact_match": "Exact match",
        "probability_full_answer": "Probability full (mean)",
    }
    logger.info(f"\n%s", "=" * 88)
    logger.info("%s", title)
    logger.info(f"%s", "=" * 88)
    logger.info(
        "%-31s %10s %10s %10s %10s",
        "Metric",
        "Context",
        "Prior",
        "Overall",
        "Pair",
    )
    logger.info("%s", "-" * 88)
    for metric_name, label in labels.items():
        values = metrics[metric_name]
        logger.info(
            "%-31s %10.2f%% %10.2f%% %10.2f%% %10.2f%%",
            label,
            100 * float(values["context_accuracy"]),
            100 * float(values["prior_accuracy"]),
            100 * float(values["overall_accuracy"]),
            100 * float(values["pair_accuracy"]),
        )
    counts = metrics["exact_match"]
    logger.info(
        "Samples=%d complete_pairs=%d incomplete_pairs=%d",
        counts["n_samples"],
        counts["n_pairs"],
        counts["n_incomplete_pairs"],
    )


# Main


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

    run = prepare_eval_run(
        category=args.eval_category,
        expid=args.expid,
        output_root=args.output_root,
        timestamp=args.timestamp,
        output_name=args.output_name,
    )

    configure_run_logging(run, kind="sft-inference")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    setup_seed(args.seed)

    device = get_device()
    logger.info(f"Using device: {device}")
    runtime = describe_runtime(device)
    logger.info("Runtime: %s", runtime)

    dataset_paths = [p.strip() for p in args.dataset_paths.split(",") if p.strip()]
    if not dataset_paths:
        raise ValueError("No dataset paths provided")
    missing_datasets = [path for path in dataset_paths if not Path(path).is_file()]
    if missing_datasets:
        raise FileNotFoundError("Evaluation datasets not found: " + ", ".join(missing_datasets))
    dataset_names = [Path(path).stem for path in dataset_paths]
    if len(set(dataset_names)) != len(set(dataset_names)):
        raise ValueError("Dataset paths must have unique filename stems for stable example IDs")

    logger.info(f"Evaluating on {len(dataset_paths)} dataset(s): {dataset_paths}")

    # Load model

    logger.info(f"Loading model from: {args.model_name_or_path}")
    logger.info("Use Image: " + str(args.use_image))

    model, processor = load_model_and_processor(
        args.model_name_or_path,
        lora_path=args.lora_name_or_path,
        merge_lora=args.lora_name_or_path is not None,
        torch_dtype=torch.bfloat16,
    )
    precision = describe_model_precision(model)
    unexpected_dtypes = set(precision["floating_parameter_counts_by_dtype"]) - {"bfloat16"}
    if unexpected_dtypes:
        raise RuntimeError(
            f"Inference requires all floating model parameters to use bfloat16, but found: {sorted(unexpected_dtypes)}"
        )
    logger.info("Inference precision: %s", precision)
    model_revision = getattr(model.config, "_commit_hash", None)
    logger.info("Model revision: %s", model_revision)
    generation_policy = describe_eval_generation(model)
    logger.info("Generation policy: %s", generation_policy)

    # Load datasets (unified pipeline with TaskMixture + CustomCollator)

    tasks: List[Task] = []
    for dataset_path in dataset_paths:
        task = Task(
            jsonl_path=dataset_path,
            image_root=args.image_root,
            processor=processor,
            max_length=args.max_length,
            stop=args.stop,  # Debug
            training=False,
            use_image=args.use_image,
            use_instruction=args.use_instruction,
        )
        tasks.append(task)
        logger.info(f"[{Path(dataset_path).stem}] Loaded {len(task)} samples")

    task_mixture = TaskMixture(tasks=tasks, shuffle=False)
    logger.info(f"Total: {len(task_mixture)} inference samples")

    collator = CustomCollator(
        processor=processor,
        max_length=args.max_length,
        inference_mode=True,
    )

    dataloader = DataLoader(
        task_mixture,  # type: ignore
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Batch inference

    results: List[dict] = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
        metadatas = batch.pop("_metadata")
        if batch_idx == 0:
            decoded_sample = processor.tokenizer.decode(batch["input_ids"][0])
            logger.info("Tokenized one sample %s", decoded_sample)

        ctx_answers = [m.context_answer.strip() for m in metadatas]
        prior_answers = [m.prior_answer.strip() for m in metadatas]
        modes = [m.mode for m in metadatas]
        image_paths = [m.image_path for m in metadatas]
        statements = [m.statement for m in metadatas]
        questions = [m.question for m in metadatas]
        datasets_names = [m.dataset_name for m in metadatas]
        response_prefixes = [m.response_prefix for m in metadatas]
        example_ids = [str(m.example_id or "") for m in metadatas]
        if any(not example_id for example_id in example_ids):
            raise ValueError("Every evaluation sample must have a stable example_id")
        with torch.inference_mode():
            predictions, logprob_scores = score_batch(
                model=model,
                inputs=batch,
                processor=processor,
                ctx_answers=ctx_answers,
                prior_answers=prior_answers,
                response_prefixes=response_prefixes,
                device=device,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
            )

        for i, (pred, lp) in enumerate(zip(predictions, logprob_scores)):
            mode = modes[i]
            expected = ctx_answers[i] if mode == "context" else prior_answers[i]
            probability_context_mean = float(lp["probability_context_logprob_mean"])
            probability_prior_mean = float(lp["probability_prior_logprob_mean"])

            probability_correct = (
                probability_context_mean > probability_prior_mean
                if mode == "context"
                else probability_prior_mean > probability_context_mean
            )
            probability_margin = (
                probability_context_mean - probability_prior_mean
                if mode == "context"
                else probability_prior_mean - probability_context_mean
            )

            results.append(
                {
                    "example_id": example_ids[i],
                    "dataset": datasets_names[i],
                    "image": image_paths[i],
                    "statement": statements[i],
                    "question": questions[i],
                    "mode": mode,
                    "response_prefix": response_prefixes[i],
                    "context_answer": ctx_answers[i],
                    "prior_answer": prior_answers[i],
                    "expected": expected,
                    "prediction": pred,
                    "expected_normalized": normalize_answer(expected),
                    "prediction_normalized": normalize_answer(pred),
                    "exact_match_correct": exact_match(pred, expected),
                    "probability_full_answer_correct": probability_correct,
                    "probability_full_answer_margin": probability_margin,
                    "probability_context_logprob_sum": float(lp["probability_context_logprob_sum"]),
                    "probability_context_logprob_mean": probability_context_mean,
                    "probability_context_token_count": int(lp["probability_context_token_count"]),
                    "probability_prior_logprob_sum": float(lp["probability_prior_logprob_sum"]),
                    "probability_prior_logprob_mean": probability_prior_mean,
                    "probability_prior_token_count": int(lp["probability_prior_token_count"]),
                }
            )

    # Compute metrics per-dataset and aggregate

    results_by_dataset: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        results_by_dataset[r["dataset"]].append(r)

    metrics_by_dataset: Dict[str, Dict[str, Dict[str, float | int]]] = {}
    for dataset_name, dataset_results in results_by_dataset.items():
        metrics_by_dataset[dataset_name] = compute_metrics(dataset_results)

    aggregate_metrics = compute_metrics(results)
    macro_metrics = compute_macro_metrics(metrics_by_dataset)

    # Save results

    output_data = {
        "config": {
            "model_name_or_path": args.model_name_or_path,
            "model_revision": model_revision,
            "lora_name_or_path": args.lora_name_or_path,
            "lora_merged": args.lora_name_or_path is not None,
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
        "metric_protocol": METRIC_PROTOCOL,
        "metrics": {
            "aggregate": aggregate_metrics,
            "macro_average": macro_metrics,
            "per_dataset": metrics_by_dataset,
        },
        "results": results,
    }
    output_path = write_json_result(run, output_data, kind="sft-inference")

    # Print summary

    logger.info(f"\nResults saved to: {output_path}")

    if len(dataset_paths) > 1:
        for dataset_name in dataset_paths:
            ds_name = Path(dataset_name).stem
            if ds_name in metrics_by_dataset:
                log_metrics(metrics_by_dataset[ds_name], title=f"Dataset: {ds_name}")
        log_metrics(macro_metrics, title="Macro-average (Unweighted)")
        log_metrics(aggregate_metrics, title="Aggregate (Micro, All Samples)")
    else:
        log_metrics(aggregate_metrics, title=f"Dataset: {Path(dataset_paths[0]).stem}")


if __name__ == "__main__":
    main()
