"""Shared full-answer probability preference evaluation utilities."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch

PREFERENCE_LABEL_NAMES = [
    "labels",
    "_candidate_is_correct",
    "_preference_group",
]


def answer_fragment(answer: str, response_prefix: str) -> str:
    """Return an answer with the boundary expected after the response prefix."""

    answer = answer.strip()
    if not answer:
        raise ValueError("Candidate answers must not be empty")
    return (" " if response_prefix.strip() else "") + answer


def candidate_token_ids(
    processor,
    answers: Sequence[str],
    response_prefixes: Sequence[str],
) -> list[list[int]]:
    """Tokenize answer fragments exactly as in the standalone SFT evaluator."""

    if len(answers) != len(response_prefixes):
        raise ValueError("answers and response_prefixes must have the same length")
    token_ids = [
        processor.tokenizer.encode(
            answer_fragment(answer, prefix),
            add_special_tokens=False,
        )
        for answer, prefix in zip(answers, response_prefixes)
    ]
    if any(not ids for ids in token_ids):
        raise ValueError("At least one candidate answer tokenized to an empty sequence")
    return token_ids


def _left_pad(
    rows: Sequence[torch.Tensor],
    *,
    padding_value: int,
) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        list(rows),
        batch_first=True,
        padding_value=padding_value,
        padding_side="left",
    )


def build_probability_inputs(
    inputs: Mapping[str, torch.Tensor],
    candidate_ids: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    max_length: int,
) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Append one candidate answer per prompt and label only its answer tokens."""

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size = input_ids.shape[0]
    if len(candidate_ids) != batch_size:
        raise ValueError("Expected one candidate answer per input row")

    combined_ids: list[torch.Tensor] = []
    combined_attention: list[torch.Tensor] = []
    combined_labels: list[torch.Tensor] = []
    sequence_fields = {
        key: []
        for key in ("mm_token_type_ids", "token_type_ids")
        if key in inputs and inputs[key].shape == input_ids.shape
    }

    for row_index, ids in enumerate(candidate_ids):
        prompt_mask = attention_mask[row_index].bool()
        prompt_ids = input_ids[row_index][prompt_mask]
        candidate = input_ids.new_tensor(ids)
        total_length = prompt_ids.numel() + candidate.numel()
        if total_length > max_length:
            raise ValueError(
                f"Probability sequence {row_index} needs {total_length} tokens, "
                f"which exceeds max_length={max_length}"
            )

        combined_ids.append(torch.cat((prompt_ids, candidate)))
        combined_attention.append(attention_mask.new_ones(total_length))
        combined_labels.append(
            torch.cat(
                (
                    input_ids.new_full((prompt_ids.numel(),), -100),
                    candidate,
                )
            )
        )
        for key, rows in sequence_fields.items():
            prompt_values = inputs[key][row_index][prompt_mask]
            answer_values = inputs[key].new_zeros(candidate.numel())
            rows.append(torch.cat((prompt_values, answer_values)))

    model_inputs = {
        key: value
        for key, value in inputs.items()
        if key
        not in {
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "token_type_ids",
        }
    }
    model_inputs["input_ids"] = _left_pad(combined_ids, padding_value=pad_token_id)
    model_inputs["attention_mask"] = _left_pad(
        combined_attention,
        padding_value=0,
    )
    for key, rows in sequence_fields.items():
        model_inputs[key] = _left_pad(rows, padding_value=0)
    labels = _left_pad(combined_labels, padding_value=-100)
    return model_inputs, labels


def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return summed and mean full-answer log-probabilities."""

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    token_counts = mask.sum(dim=1)
    if torch.any(token_counts == 0):
        raise ValueError("At least one candidate answer contains no scoreable tokens")

    selected_logits = shifted_logits[mask].float()
    selected_labels = shifted_labels[mask]
    selected_logprobs = torch.log_softmax(selected_logits, dim=-1).gather(
        dim=-1,
        index=selected_labels.unsqueeze(-1),
    )
    batch_indices = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(mask)[mask]
    logprob_sums = selected_logprobs.new_zeros(labels.shape[0])
    logprob_sums.scatter_add_(0, batch_indices, selected_logprobs.squeeze(-1))
    return logprob_sums, logprob_sums / token_counts, token_counts


class PreferenceDataset:
    """Expand each validation prompt into correct and alternative candidates."""

    def __init__(
        self,
        prompt_dataset,
        processor,
        *,
        max_length: int,
    ) -> None:
        self.prompt_dataset = prompt_dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return 2 * len(self.prompt_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not isinstance(index, int):
            raise TypeError(f"index must be an integer, got {type(index)}")
        prompt_index, candidate_index = divmod(index, 2)
        item = dict(self.prompt_dataset[prompt_index])
        metadata = item.pop("metadata")
        candidate_is_correct = candidate_index == 0

        correct_answer = metadata.context_answer if metadata.mode == "context" else metadata.prior_answer
        alternative_answer = metadata.prior_answer if metadata.mode == "context" else metadata.context_answer
        answer = correct_answer if candidate_is_correct else alternative_answer
        ids = candidate_token_ids(
            self.processor,
            [answer],
            [metadata.response_prefix],
        )[0]

        input_ids = item["input_ids"]
        candidate = input_ids.new_tensor(ids)
        total_length = input_ids.numel() + candidate.numel()
        if total_length > self.max_length:
            raise ValueError(
                f"Probability sequence {prompt_index} needs {total_length} "
                f"tokens, which exceeds max_length={self.max_length}"
            )

        if (
            "gemma" in self.processor.tokenizer.name_or_path.lower()
            and "mm_token_type_ids" in item
            and "token_type_ids" not in item
        ):
            item["token_type_ids"] = item["mm_token_type_ids"].clone()

        item["input_ids"] = torch.cat((input_ids, candidate))
        item["attention_mask"] = torch.cat((item["attention_mask"], item["attention_mask"].new_ones(candidate.numel())))
        for key in ("mm_token_type_ids", "token_type_ids"):
            if key in item:
                item[key] = torch.cat((item[key], item[key].new_zeros(candidate.numel())))
        item["labels"] = torch.cat(
            (
                input_ids.new_full((input_ids.numel(),), -100),
                candidate,
            )
        )
        item["_candidate_is_correct"] = torch.tensor(
            [candidate_is_correct],
            dtype=torch.bool,
        )
        item["_preference_group"] = torch.tensor([prompt_index], dtype=torch.long)
        return item


def preprocess_logits_for_preference(
    logits: torch.Tensor | tuple,
    labels: torch.Tensor | tuple,
) -> torch.Tensor:
    """Reduce full logits to one mean answer log-probability per candidate."""

    if isinstance(logits, tuple):
        logits = logits[0]
    sequence_labels = labels[0] if isinstance(labels, tuple) else labels
    _, means, _ = sequence_logprobs(logits, sequence_labels)
    return means.unsqueeze(-1)


def compute_probability_preference_metrics(eval_prediction) -> dict[str, float]:
    """Compare the correct and alternative full-answer mean log-probabilities."""

    if not isinstance(eval_prediction.label_ids, (tuple, list)):
        raise ValueError("Preference evaluation requires candidate and group labels")
    if len(eval_prediction.label_ids) != len(PREFERENCE_LABEL_NAMES):
        raise ValueError("Unexpected number of preference label tensors")

    scores = np.asarray(eval_prediction.predictions).reshape(-1)
    candidate_is_correct = np.asarray(eval_prediction.label_ids[1]).astype(bool).reshape(-1)
    group_ids = np.asarray(eval_prediction.label_ids[2]).reshape(-1)
    if not (len(scores) == len(candidate_is_correct) == len(group_ids)):
        raise ValueError("Preference predictions and labels have different lengths")

    prompt_correct: dict[int, bool] = {}
    margins: list[float] = []
    for raw_group_id in np.unique(group_ids):
        group_id = int(raw_group_id)
        selected = group_ids == raw_group_id
        selected_scores = scores[selected]
        selected_targets = candidate_is_correct[selected]
        if selected_scores.size != 2 or int(selected_targets.sum()) != 1:
            raise ValueError(f"Preference group {group_id} must contain one correct and one alternative candidate")
        correct_score = float(selected_scores[selected_targets][0])
        alternative_score = float(selected_scores[~selected_targets][0])
        margin = correct_score - alternative_score
        margins.append(margin)
        prompt_correct[group_id] = margin > 0.0

    if not prompt_correct:
        return {
            "accuracy": 0.0,
            "context_accuracy": 0.0,
            "prior_accuracy": 0.0,
            "pair_accuracy": 0.0,
            "min_accuracy": 0.0,
            "mean_margin": 0.0,
        }

    context_results = [correct for group_id, correct in prompt_correct.items() if group_id % 2 == 0]
    prior_results = [correct for group_id, correct in prompt_correct.items() if group_id % 2 == 1]
    complete_example_ids = {
        group_id // 2
        for group_id in prompt_correct
        if 2 * (group_id // 2) in prompt_correct and 2 * (group_id // 2) + 1 in prompt_correct
    }
    pair_results = [
        prompt_correct[2 * example_id] and prompt_correct[2 * example_id + 1] for example_id in complete_example_ids
    ]

    def average(values: Sequence[bool]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "accuracy": average(list(prompt_correct.values())),
        "context_accuracy": average(context_results),
        "prior_accuracy": average(prior_results),
        "pair_accuracy": average(pair_results),
        "min_accuracy": min(
            average(context_results),
            average(prior_results),
        ),
        "mean_margin": float(np.mean(margins)),
    }


class BaselineRelativePreferenceMetrics:
    """Add strict two-branch gains to the standard preference metrics."""

    def __init__(self) -> None:
        self.baseline_context_accuracy: float | None = None
        self.baseline_prior_accuracy: float | None = None

    def set_baseline(self, *, context_accuracy: float, prior_accuracy: float) -> None:
        self.baseline_context_accuracy = float(context_accuracy)
        self.baseline_prior_accuracy = float(prior_accuracy)

    def clear_baseline(self) -> None:
        self.baseline_context_accuracy = None
        self.baseline_prior_accuracy = None

    def _add_baseline_metrics(
        self,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        if self.baseline_context_accuracy is None or self.baseline_prior_accuracy is None:
            return metrics

        context_gain = metrics["context_accuracy"] - self.baseline_context_accuracy
        prior_gain = metrics["prior_accuracy"] - self.baseline_prior_accuracy
        metrics.update(
            {
                "context_gain": context_gain,
                "prior_gain": prior_gain,
                "joint_gain": min(context_gain, prior_gain),
                "both_improved": float(context_gain > 0.0 and prior_gain > 0.0),
            }
        )
        return metrics

    def __call__(self, eval_prediction) -> dict[str, float]:
        return self._add_baseline_metrics(compute_probability_preference_metrics(eval_prediction))


class ValidationSuitePreferenceMetrics(BaselineRelativePreferenceMetrics):
    """Compute equally weighted aggregate and per-split validation metrics.

    The combined validation dataset is laid out as contiguous prompt ranges.
    Computing every split from the gathered predictions avoids five additional
    evaluation passes per epoch. Top-level metrics are macro averages across
    splits, so a large validation split cannot dominate checkpoint selection.
    """

    def __init__(self, split_prompt_counts: Sequence[tuple[str, int]]) -> None:
        super().__init__()
        if not split_prompt_counts:
            raise ValueError("split_prompt_counts must not be empty")
        names = [name for name, _ in split_prompt_counts]
        if len(names) != len(set(names)):
            raise ValueError("Validation split names must be unique")
        if any(not name or count <= 0 for name, count in split_prompt_counts):
            raise ValueError("Validation split names and prompt counts must be valid")
        self.split_prompt_counts = list(split_prompt_counts)

    def __call__(self, eval_prediction) -> dict[str, float]:
        if not isinstance(eval_prediction.label_ids, (tuple, list)):
            raise ValueError("Preference evaluation requires candidate and group labels")
        if len(eval_prediction.label_ids) != len(PREFERENCE_LABEL_NAMES):
            raise ValueError("Unexpected number of preference label tensors")

        predictions = np.asarray(eval_prediction.predictions)
        labels = tuple(np.asarray(value) for value in eval_prediction.label_ids)
        group_ids = labels[2].reshape(-1)
        expected_prompt_count = sum(count for _, count in self.split_prompt_counts)
        observed_group_ids = np.unique(group_ids)
        if not np.array_equal(
            observed_group_ids,
            np.arange(expected_prompt_count, dtype=observed_group_ids.dtype),
        ):
            raise ValueError(
                "Validation suite preference groups do not match configured split "
                f"sizes: expected 0..{expected_prompt_count - 1}"
            )

        split_metrics: dict[str, dict[str, float]] = {}
        start = 0
        for split_name, prompt_count in self.split_prompt_counts:
            stop = start + prompt_count
            selected = (group_ids >= start) & (group_ids < stop)
            split_prediction = type(
                "PreferencePrediction",
                (),
                {
                    "predictions": predictions[selected],
                    "label_ids": (
                        labels[0][selected],
                        labels[1][selected],
                        labels[2][selected] - start,
                    ),
                },
            )()
            split_metrics[split_name] = compute_probability_preference_metrics(split_prediction)
            start = stop

        aggregate_keys = (
            "accuracy",
            "context_accuracy",
            "prior_accuracy",
            "pair_accuracy",
            "mean_margin",
        )
        metrics = {key: float(np.mean([values[key] for values in split_metrics.values()])) for key in aggregate_keys}

        metrics["min_accuracy"] = min(
            metrics["context_accuracy"],
            metrics["prior_accuracy"],
        )
        for split_name, values in split_metrics.items():
            metrics.update({f"{split_name}_{key}": value for key, value in values.items()})
        return self._add_baseline_metrics(metrics)
