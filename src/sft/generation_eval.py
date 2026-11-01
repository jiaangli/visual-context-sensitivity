"""Generation-based exact-match evaluation shared by training entry points."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.eval_generation import build_eval_generation_kwargs, describe_eval_generation
from src.sft.data_process import CustomCollator

EXACT_MATCH_PROTOCOL = {
    "normalization": "NFKC, casefold, replace non-letter/digit runs with spaces",
    "comparison": "full-string equality",
    "pair": "both context and prior prompts must be correct for one example_id",
}


def normalize_answer(text: str) -> str:
    """Normalize an answer without changing its word-level meaning."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        character if unicodedata.category(character)[0] in {"L", "N"} else " " for character in normalized
    )
    return " ".join(normalized.split())


def exact_match(prediction: str, expected: str) -> bool:
    """Return whether the complete normalized prediction equals the answer."""

    normalized_expected = normalize_answer(expected)
    return bool(normalized_expected) and normalize_answer(prediction) == normalized_expected


def compute_exact_match_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Compute context, prior, overall, and paired exact-match accuracies."""

    context_rows = [row for row in rows if row["mode"] == "context"]
    prior_rows = [row for row in rows if row["mode"] == "prior"]

    def accuracy(selected_rows: Sequence[Mapping[str, Any]]) -> float:
        if not selected_rows:
            return 0.0
        return sum(bool(row["exact_match_correct"]) for row in selected_rows) / len(selected_rows)

    paired: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        mode = str(row["mode"])
        if mode not in {"context", "prior"}:
            raise ValueError(f"Unknown evaluation mode: {mode!r}")
        example_id = str(row.get("example_id") or "")
        if not example_id:
            raise ValueError("Every result row must contain a non-empty example_id")
        pair = paired.setdefault(example_id, {})
        if mode in pair:
            raise ValueError(f"Duplicate {mode} result for example_id={example_id}")
        pair[mode] = row

    complete_pairs = [pair for pair in paired.values() if {"context", "prior"} <= pair.keys()]
    pair_accuracy = (
        sum(
            bool(pair["context"]["exact_match_correct"]) and bool(pair["prior"]["exact_match_correct"])
            for pair in complete_pairs
        )
        / len(complete_pairs)
        if complete_pairs
        else 0.0
    )
    return {
        "context_accuracy": accuracy(context_rows),
        "prior_accuracy": accuracy(prior_rows),
        "overall_accuracy": accuracy(rows),
        "pair_accuracy": pair_accuracy,
        "n_samples": len(rows),
        "n_pairs": len(complete_pairs),
        "n_incomplete_pairs": len(paired) - len(complete_pairs),
    }


def summarize_exact_match(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return per-dataset, aggregate, and unweighted dataset-macro EM."""

    rows_by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_dataset[str(row["dataset"])].append(row)
    per_dataset = {name: compute_exact_match_metrics(dataset_rows) for name, dataset_rows in rows_by_dataset.items()}
    aggregate = compute_exact_match_metrics(rows)
    accuracy_keys = (
        "context_accuracy",
        "prior_accuracy",
        "overall_accuracy",
        "pair_accuracy",
    )
    count_keys = ("n_samples", "n_pairs", "n_incomplete_pairs")
    macro_average: dict[str, float | int] = {
        key: (sum(float(metrics[key]) for metrics in per_dataset.values()) / len(per_dataset) if per_dataset else 0.0)
        for key in accuracy_keys
    }
    macro_average.update({key: sum(int(metrics[key]) for metrics in per_dataset.values()) for key in count_keys})
    return {
        "protocol": EXACT_MATCH_PROTOCOL,
        "aggregate": aggregate,
        "macro_average": macro_average,
        "per_dataset": per_dataset,
    }


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate_exact_match_generation(
    *,
    model,
    processor,
    dataset,
    device: str | torch.device,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Generate answers once and calculate exact-match metrics.

    The returned summary includes per-sample ``results`` rows (prediction,
    expected answer, normalized forms, correctness) in dataset order.
    """

    if not dataset:
        raise ValueError("Generation evaluation dataset is empty")
    if batch_size <= 0:
        raise ValueError("Generation evaluation batch_size must be positive")
    if max_new_tokens <= 0:
        raise ValueError("Generation evaluation max_new_tokens must be positive")

    resolved_device = torch.device(device)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": CustomCollator(
            processor=processor,
            max_length=max_length,
            inference_mode=True,
        ),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": resolved_device.type == "cuda",
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    dataloader = DataLoader(dataset, **loader_kwargs)

    rows: list[dict[str, Any]] = []
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer defines neither pad_token_id nor eos_token_id")
    previous_padding_side = tokenizer.padding_side
    was_training = model.training
    tokenizer.padding_side = "left"
    model.eval()
    try:
        for batch in tqdm(
            dataloader,
            desc="Generating final-test EM",
            disable=not show_progress,
        ):
            metadata = batch.pop("_metadata")
            batch.pop("labels", None)
            inputs = {
                key: value.to(resolved_device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            if "gemma" in tokenizer.name_or_path.lower() and "mm_token_type_ids" in inputs:
                inputs["token_type_ids"] = inputs["mm_token_type_ids"]
            input_ids = inputs["input_ids"]
            generation_kwargs = build_eval_generation_kwargs(model, input_ids)
            with torch.inference_mode(), _autocast(resolved_device):
                output = model.generate(
                    **inputs,
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    **generation_kwargs,
                )
            sequences = getattr(output, "sequences", output)
            generated_tokens = sequences[:, input_ids.shape[1] :]
            predictions = processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            for prediction, sample in zip(predictions, metadata, strict=True):
                expected = sample.context_answer if sample.mode == "context" else sample.prior_answer
                rows.append(
                    {
                        "example_id": str(sample.example_id or ""),
                        "dataset": sample.dataset_name,
                        "image": sample.image_path,
                        "statement": sample.statement,
                        "question": sample.question,
                        "mode": sample.mode,
                        "response_prefix": sample.response_prefix,
                        "context_answer": sample.context_answer,
                        "prior_answer": sample.prior_answer,
                        "expected": expected,
                        "prediction": prediction,
                        "expected_normalized": normalize_answer(expected),
                        "prediction_normalized": normalize_answer(prediction),
                        "exact_match_correct": exact_match(prediction, expected),
                    }
                )
    finally:
        tokenizer.padding_side = previous_padding_side
        model.train(was_training)

    summary = summarize_exact_match(rows)
    summary["config"] = {
        "batch_size": batch_size,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "generation": describe_eval_generation(model),
    }
    summary["results"] = rows
    return summary
