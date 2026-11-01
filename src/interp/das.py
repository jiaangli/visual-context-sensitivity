import logging
import statistics
import warnings
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from nnsight import NNsight
from torch import Tensor
from torch.nn import CrossEntropyLoss
from tqdm import trange

from src.interp.api.qwen2p5vl import Qwen2p5VL
from src.interp.decoding import (
    config_to_site,
    get_num_layers,
    get_patched_residuals,
    get_probs,
    patch_scope,
)
from src.interp.layer_selection import (
    DEFAULT_ONSET_PERSISTENCE,
    DEFAULT_ONSET_THRESHOLD,
    annotate_difference_onset,
)
from src.sft.preference_eval import build_probability_inputs

logger = logging.getLogger(__name__)


def compute_loss(logits: Tensor, labels: Tensor) -> Tensor:
    last_token_logits = logits[:, -1, :]
    loss_fct = CrossEntropyLoss()
    labels = labels.to(logits.device)
    return loss_fct(last_token_logits, labels[:, 0])


def compute_metrics(
    eval_preds: List[Tensor],
    eval_labels: List[Tensor],
    tokenizer=None,
    return_target_accuracy: bool = True,
    verbose: bool = False,
) -> Dict[str, float]:
    total_count = 0
    correct_count = 0
    alter_correct_count = 0

    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        len_before = len(eval_label)
        eval_pred = eval_pred[eval_label[:, 0] != eval_label[:, 1]]
        eval_label = eval_label[eval_label[:, 0] != eval_label[:, 1]]
        len_after = len(eval_label)

        if len_before != len_after:
            logger.info(f"Filtered {len_before - len_after} samples due to equal first token")

        pred_test_labels = torch.argmax(eval_pred[:, -1, :], dim=-1)
        correct_labels = eval_label[:, 0] == pred_test_labels
        alter_correct_count += (eval_label[:, 1] == pred_test_labels).sum().tolist()

        if verbose and tokenizer is not None:
            for b_idx in range(len(eval_pred)):
                print(
                    "Pred:",
                    tokenizer.decode(pred_test_labels[b_idx].item()),
                    " Labels (source):",
                    tokenizer.decode(eval_label[b_idx][0].item()),
                    " Labels (target):",
                    tokenizer.decode(eval_label[b_idx][1].item()),
                )
                print("Correct:", correct_labels[b_idx].item())
                print(
                    "Alter Correct:",
                    (eval_label[b_idx][1] == pred_test_labels[b_idx]).item(),
                )

        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()

    accuracy = round(correct_count / total_count, 2) if total_count > 0 else 0.0
    alter_accuracy = round(alter_correct_count / total_count, 2) if total_count > 0 else 0.0

    if return_target_accuracy:
        return {"accuracy": accuracy, "target_accuracy": alter_accuracy}
    return {"accuracy": accuracy}


def get_patch_scope_probs(
    nnmodel: NNsight,
    processor,
    source_input_ids: Tensor,
    target_input_ids: Tensor,
    source_attention_mask: Tensor,
    target_attention_mask: Tensor,
    source_pixel_values: Tensor,
    target_pixel_values: Tensor,
    source_image_grid_thw: Tensor,
    target_image_grid_thw: Tensor,
    source_image_position_ids: Tensor,
    target_image_position_ids: Tensor,
    source_mm_token_type_ids: Tensor,
    target_mm_token_type_ids: Tensor,
    source_answer: Tensor,
    target_answer: Tensor,
    site,
    batch_size: int = 16,
    api=Qwen2p5VL,
    max_samples: Optional[int] = None,
    return_stats: bool = False,
) -> Tuple[Tensor, ...]:
    """
    Compute patch scope probabilities following the reference implementation.

    Instead of accepting a pre-built site, this now takes a site_config dict
    like {"o": {"layers": [0,1,2]}} and builds the site internally.
    """
    pixel_value_dim = source_pixel_values.size(-1) if source_pixel_values is not None else None
    if max_samples is not None:
        target_input_ids = target_input_ids[:max_samples]
        source_input_ids = source_input_ids[:max_samples]
        source_attention_mask = source_attention_mask[:max_samples]
        target_attention_mask = target_attention_mask[:max_samples]
        source_answer = source_answer[:max_samples]
        target_answer = target_answer[:max_samples]
        if source_pixel_values is not None:
            source_pixel_values = source_pixel_values[:max_samples]
            target_pixel_values = target_pixel_values[:max_samples]
            if source_image_grid_thw is not None:
                source_image_grid_thw = source_image_grid_thw[:max_samples]
                target_image_grid_thw = target_image_grid_thw[:max_samples]

        if source_image_position_ids is not None:
            source_image_position_ids = source_image_position_ids[:max_samples]
            target_image_position_ids = target_image_position_ids[:max_samples]
        if source_mm_token_type_ids is not None:
            source_mm_token_type_ids = source_mm_token_type_ids[:max_samples]
            target_mm_token_type_ids = target_mm_token_type_ids[:max_samples]

    if target_input_ids.shape[0] == 0:
        raise ValueError("No samples to process after filtering. Please check your inputs or filtering criteria.")

    input_device = getattr(nnmodel, "device", source_input_ids.device)

    residuals = []
    for i in range(0, target_input_ids.shape[0], batch_size):
        site.reset()
        if source_pixel_values is not None:
            if source_image_grid_thw is not None:
                src_grid = source_image_grid_thw[i : i + batch_size].to(input_device)
                tgt_grid = target_image_grid_thw[i : i + batch_size].to(input_device)
            else:
                src_grid = None
                tgt_grid = None
            src_pv = source_pixel_values[i : i + batch_size].to(input_device)
            tgt_pv = target_pixel_values[i : i + batch_size].to(input_device)
        else:
            src_pv = None
            tgt_pv = None
            src_grid = None
            tgt_grid = None
        if source_image_position_ids is not None:
            src_img_pos_ids = source_image_position_ids[i : i + batch_size].to(input_device)
            tgt_img_pos_ids = target_image_position_ids[i : i + batch_size].to(input_device)
        else:
            src_img_pos_ids = None
            tgt_img_pos_ids = None
        if source_mm_token_type_ids is not None:
            src_mm_ids = source_mm_token_type_ids[i : i + batch_size].to(input_device)
            tgt_mm_ids = target_mm_token_type_ids[i : i + batch_size].to(input_device)
        else:
            src_mm_ids = None
            tgt_mm_ids = None

        residuals_batch, _ = get_patched_residuals(
            nnmodel,
            site,
            source_input_ids[i : i + batch_size].to(input_device),
            target_input_ids[i : i + batch_size].to(input_device),
            source_attention_mask[i : i + batch_size].to(input_device),
            target_attention_mask[i : i + batch_size].to(input_device),
            source_pixel_values=src_pv
            if (src_pv is not None and src_img_pos_ids is not None)
            else (src_pv.view(-1, pixel_value_dim) if src_pv is not None else None),
            target_pixel_values=tgt_pv
            if (tgt_pv is not None and tgt_img_pos_ids is not None)
            else (tgt_pv.view(-1, pixel_value_dim) if tgt_pv is not None else None),
            source_image_grid_thw=src_grid if src_grid is not None else None,
            target_image_grid_thw=tgt_grid if tgt_grid is not None else None,
            source_image_position_ids=src_img_pos_ids if src_img_pos_ids is not None else None,
            target_image_position_ids=tgt_img_pos_ids if tgt_img_pos_ids is not None else None,
            source_mm_token_type_ids=src_mm_ids if src_mm_ids is not None else None,
            target_mm_token_type_ids=tgt_mm_ids if tgt_mm_ids is not None else None,
            scan=False,
            validate=False,
        )

        residuals.append(residuals_batch)

    residuals = torch.cat(residuals, dim=1)

    logits = patch_scope(nnmodel, processor, residuals, verbose=False)

    source_stats = get_probs(logits, source_answer.to(input_device))
    target_stats = get_probs(logits, target_answer.to(input_device))

    if return_stats:
        return source_stats, target_stats

    return source_stats[0], target_stats[0]


def _prefixed_side_inputs(
    patching_arguments,
    side: str,
    row_indices: Sequence[int],
    prefixes: Sequence[Sequence[int]],
    *,
    processor,
    max_length: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Build a left-padded prompt plus a shared candidate-answer prefix."""
    if side not in {"source", "target"}:
        raise ValueError(f"side must be 'source' or 'target', got {side!r}")
    if len(row_indices) != len(prefixes):
        raise ValueError("row_indices and prefixes must have the same length")

    def select(value: Tensor) -> Tensor:
        index = torch.tensor(row_indices, dtype=torch.long, device=value.device)
        return value[index]

    prompt_inputs: Dict[str, Tensor] = {
        "input_ids": select(getattr(patching_arguments, f"{side}_input_ids")),
        "attention_mask": select(getattr(patching_arguments, f"{side}_attention_mask")),
    }
    for name in (
        "pixel_values",
        "image_grid_thw",
        "image_position_ids",
        "mm_token_type_ids",
    ):
        value = getattr(patching_arguments, f"{side}_{name}")
        if value is not None:
            prompt_inputs[name] = select(value)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("The tokenizer needs a pad_token_id or eos_token_id")

    model_inputs, _ = build_probability_inputs(
        prompt_inputs,
        prefixes,
        pad_token_id=pad_token_id,
        max_length=max_length,
    )
    model_inputs = {key: value.to(device, non_blocking=True) for key, value in model_inputs.items()}
    if "pixel_values" in model_inputs and "image_position_ids" not in model_inputs:
        pixel_values = model_inputs["pixel_values"]
        if pixel_values.dim() == 3:
            model_inputs["pixel_values"] = pixel_values.view(-1, pixel_values.size(-1))
    return model_inputs


def _patched_next_token_logits(
    nnmodel: NNsight,
    site,
    source_inputs: Dict[str, Tensor],
    target_inputs: Dict[str, Tensor],
) -> Tensor:
    """Patch the shared last-prefix position and return next-token logits."""
    _, logits = get_patched_residuals(
        nnmodel,
        site,
        source_inputs["input_ids"],
        target_inputs["input_ids"],
        source_inputs["attention_mask"],
        target_inputs["attention_mask"],
        source_pixel_values=source_inputs.get("pixel_values"),
        target_pixel_values=target_inputs.get("pixel_values"),
        source_image_grid_thw=source_inputs.get("image_grid_thw"),
        target_image_grid_thw=target_inputs.get("image_grid_thw"),
        source_image_position_ids=source_inputs.get("image_position_ids"),
        target_image_position_ids=target_inputs.get("image_position_ids"),
        source_mm_token_type_ids=source_inputs.get("mm_token_type_ids"),
        target_mm_token_type_ids=target_inputs.get("mm_token_type_ids"),
        scan=False,
        validate=False,
        return_logits=True,
        return_residuals=False,
    )
    if logits is None:
        raise RuntimeError("Patched forward pass did not return logits")
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 2:
        raise RuntimeError(f"Expected patched next-token logits with rank 2, got {tuple(logits.shape)}")
    return logits


def get_patched_full_answer_scores(
    nnmodel: NNsight,
    processor,
    patching_arguments,
    source_answer_ids: Sequence[Sequence[int]],
    target_answer_ids: Sequence[Sequence[int]],
    site,
    *,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    max_length: int = 512,
) -> Dict[str, Any]:
    """Score complete source/target answers under last-position patching.

    For answer token ``y_t``, both source and target prompts receive the same
    candidate-answer prefix ``y_<t``. The source activation at that
    last prefix position is patched into the target prompt, and ``y_t`` is
    scored from the patched target logits. Repeating this for every token is
    the autoregressive intervention counterpart of full-answer scoring.
    """
    n_rows = int(patching_arguments.source_input_ids.shape[0])
    if len(source_answer_ids) != n_rows or len(target_answer_ids) != n_rows:
        raise ValueError("Full-answer token lists must align with PatchingData rows")
    if max_samples is not None:
        n_rows = min(n_rows, max_samples)
    if n_rows == 0:
        raise ValueError("Layer search requires at least one input pair")

    source_answer_ids = [list(ids) for ids in source_answer_ids[:n_rows]]
    target_answer_ids = [list(ids) for ids in target_answer_ids[:n_rows]]
    if any(not ids for ids in source_answer_ids + target_answer_ids):
        raise ValueError("Full-answer token lists must not be empty")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    input_device = torch.device(getattr(nnmodel, "device", patching_arguments.source_input_ids.device))

    def score_candidates(candidate_ids: Sequence[Sequence[int]]):
        score_sums = torch.zeros(n_rows, dtype=torch.float64)
        token_counts = torch.zeros(n_rows, dtype=torch.long)
        patched_calls = 0
        for batch_start in range(0, n_rows, batch_size):
            batch_end = min(batch_start + batch_size, n_rows)
            max_tokens = max(len(candidate_ids[index]) for index in range(batch_start, batch_end))
            for token_index in range(max_tokens):
                active_rows = [
                    index for index in range(batch_start, batch_end) if token_index < len(candidate_ids[index])
                ]
                prefixes = [candidate_ids[index][:token_index] for index in active_rows]
                source_inputs = _prefixed_side_inputs(
                    patching_arguments,
                    "source",
                    active_rows,
                    prefixes,
                    processor=processor,
                    max_length=max_length,
                    device=input_device,
                )
                target_inputs = _prefixed_side_inputs(
                    patching_arguments,
                    "target",
                    active_rows,
                    prefixes,
                    processor=processor,
                    max_length=max_length,
                    device=input_device,
                )
                logits = _patched_next_token_logits(
                    nnmodel,
                    site,
                    source_inputs,
                    target_inputs,
                )
                labels = torch.tensor(
                    [candidate_ids[index][token_index] for index in active_rows],
                    dtype=torch.long,
                    device=logits.device,
                )
                token_logprobs = (
                    torch.log_softmax(logits.float(), dim=-1)
                    .gather(-1, labels.unsqueeze(-1))
                    .squeeze(-1)
                    .detach()
                    .double()
                    .cpu()
                )
                score_sums[active_rows] += token_logprobs
                token_counts[active_rows] += 1
                patched_calls += 1

        if torch.any(token_counts == 0):
            raise RuntimeError("At least one complete answer received no token scores")
        return score_sums / token_counts, token_counts, patched_calls

    source_scores, source_counts, source_calls = score_candidates(source_answer_ids)
    target_scores, target_counts, target_calls = score_candidates(target_answer_ids)
    margins = source_scores - target_scores
    return {
        "source_mean_logprob": float(source_scores.mean().item()),
        "target_mean_logprob": float(target_scores.mean().item()),
        "margin": float(margins.mean().item()),
        "source_per_sample": source_scores.tolist(),
        "target_per_sample": target_scores.tolist(),
        "margin_per_sample": margins.tolist(),
        "source_token_counts": source_counts.tolist(),
        "target_token_counts": target_counts.tolist(),
        "patched_forward_pairs": source_calls + target_calls,
        "model_forward_count": 2 * (source_calls + target_calls),
        "n_samples": n_rows,
    }


def _probability_stats_to_trace(stats: Tuple[Tensor, Tensor, Tensor]) -> Dict[str, List[float]]:
    """Convert PatchScope probability statistics to JSON-safe layer vectors."""
    names = ("mean", "std", "median")
    return {name: tensor.detach().float().cpu().reshape(-1).tolist() for name, tensor in zip(names, stats)}


def collect_patch_scope_gap_data(
    nnmodel: NNsight,
    processor,
    patching_arguments,
    patch_layers: Sequence[int],
    *,
    model,
    api=Qwen2p5VL,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect the inputs needed by the probability-gap figure and onset selection."""
    common_kwargs = {
        "nnmodel": nnmodel,
        "processor": processor,
        "source_input_ids": patching_arguments.source_input_ids,
        "target_input_ids": patching_arguments.target_input_ids,
        "source_attention_mask": patching_arguments.source_attention_mask,
        "target_attention_mask": patching_arguments.target_attention_mask,
        "source_mm_token_type_ids": patching_arguments.source_mm_token_type_ids,
        "target_mm_token_type_ids": patching_arguments.target_mm_token_type_ids,
        "source_answer": patching_arguments.source_answer_index,
        "target_answer": patching_arguments.target_answer_index,
        "source_pixel_values": patching_arguments.source_pixel_values,
        "target_pixel_values": patching_arguments.target_pixel_values,
        "source_image_grid_thw": patching_arguments.source_image_grid_thw,
        "target_image_grid_thw": patching_arguments.target_image_grid_thw,
        "source_image_position_ids": patching_arguments.source_image_position_ids,
        "target_image_position_ids": patching_arguments.target_image_position_ids,
        "batch_size": batch_size,
        "max_samples": max_samples,
    }

    def evaluate(layers: Sequence[int]) -> Dict[str, Any]:
        site = config_to_site({"o": {"layers": list(layers)}}, api=api, model=model)
        source_stats, target_stats = get_patch_scope_probs(
            **common_kwargs,
            site=site,
            return_stats=True,
        )
        return {
            "source": _probability_stats_to_trace(source_stats),
            "target": _probability_stats_to_trace(target_stats),
        }

    return {
        "schema_version": 1,
        "metric": "first_token_patchscope_probability_gap",
        "patch_layers": list(patch_layers),
        "n_samples": min(
            int(patching_arguments.source_input_ids.shape[0]),
            max_samples if max_samples is not None else int(patching_arguments.source_input_ids.shape[0]),
        ),
        "baseline": evaluate([]),
        "patched": evaluate(patch_layers),
    }


def _modified_zscores(values):
    if not values:
        return []
    med = statistics.median(values)
    mad = statistics.median([abs(x - med) for x in values])
    if mad == 0:
        return [0.0] * len(values)
    return [0.6745 * (x - med) / mad for x in values]


def _combine_early_and_tail_layers(
    early_positions: List[int],
    layer_indices: List[int],
    lower_bound: int,
    upper_bound: int,
) -> List[int]:
    """Convert search positions to model-layer ids before building a result."""
    early_layers = [layer_indices[position] for position in early_positions]
    return early_layers + layer_indices[lower_bound:upper_bound]


def _find_refinement_candidate(
    p_mean_source: Tensor,
    layer_indices: List[int],
    base_range: List[int],
    phi: float,
) -> Optional[int]:
    """Return the next actual layer whose probe probability has a cliff."""
    max_base_layer = max(base_range)
    start_pos = next(
        (position for position, layer in enumerate(layer_indices) if layer > max_base_layer),
        len(layer_indices),
    )
    for position in range(start_pos, len(layer_indices)):
        layer = layer_indices[position]
        previous_layer = layer_indices[position - 1]
        if p_mean_source[previous_layer, 0].item() - p_mean_source[layer, 0].item() > phi and layer not in base_range:
            return layer
    return None


def auto_search_full_answer(
    model,
    processor,
    patching_arguments,
    source_answer_ids: Sequence[Sequence[int]],
    target_answer_ids: Sequence[Sequence[int]],
    n_layers: Optional[int] = None,
    retention_threshold: float = 0.8,
    batch_size: int = 16,
    api=Qwen2p5VL,
    max_samples: Optional[int] = None,
    max_length: int = 512,
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
    search_trace: Optional[MutableMapping[str, Any]] = None,
    trace_callback: Optional[Callable[[MutableMapping[str, Any]], None]] = None,
    minimum_full_effect: float = 1e-8,
    collect_plot_data: bool = False,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    onset_persistence: int = DEFAULT_ONSET_PERSISTENCE,
) -> List[int]:
    """Find a compact contiguous range by normalized full-answer effect.

    ``m(S)`` is the mean source-vs-target complete-answer log-probability
    margin after patching layer set ``S``. The searched quantity is
    ``R(S)=(m(S)-m(empty))/(m(all)-m(empty))``. A prefix search locates the
    upper boundary, then a suffix trim locates the lower boundary. Both use
    the same ``retention_threshold`` and assume approximately monotone effects.
    """
    if n_layers is None:
        n_layers = get_num_layers(model)
    if not 0 < retention_threshold <= 1:
        raise ValueError(f"retention_threshold must be in (0, 1], got {retention_threshold}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if not 0 < onset_threshold <= 2:
        raise ValueError(f"onset_threshold must be in (0, 2], got {onset_threshold}")
    if isinstance(onset_persistence, bool) or not isinstance(onset_persistence, int) or onset_persistence <= 0:
        raise ValueError(f"onset_persistence must be a positive integer, got {onset_persistence!r}")

    if layer_indices is None:
        layer_indices = list(range(n_layers))
    else:
        layer_indices = list(layer_indices)
    if not layer_indices:
        raise ValueError("Layer search requires at least one searchable layer")
    if layer_indices != sorted(set(layer_indices)):
        raise ValueError("layer_indices must be strictly increasing and unique")
    if layer_indices[0] < 0 or layer_indices[-1] >= n_layers:
        raise ValueError(f"layer_indices must be within [0, {n_layers - 1}], got {layer_indices}")
    n_search = len(layer_indices)
    if upper_bound is not None and not 1 <= upper_bound <= n_search:
        raise ValueError(f"upper_bound is an exclusive position and must be in [1, {n_search}]")
    if lower_bound is not None and not 0 <= lower_bound < n_search:
        raise ValueError(f"lower_bound is a position and must be in [0, {n_search - 1}]")

    n_rows = int(patching_arguments.source_input_ids.shape[0])
    if len(source_answer_ids) != n_rows or len(target_answer_ids) != n_rows:
        raise ValueError("Full-answer token lists must align with PatchingData rows")

    nnmodel = NNsight(model)
    evaluation_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    if search_trace is not None:
        search_trace.clear()
        search_trace.update(
            {
                "schema_version": 2,
                "status": "running",
                "objective": "full_answer_logprob_margin",
                "n_layers": n_layers,
                "searchable_layers": list(layer_indices),
                "parameters": {
                    "retention_threshold": retention_threshold,
                    "batch_size": batch_size,
                    "max_samples": max_samples,
                    "max_length": max_length,
                    "minimum_full_effect": minimum_full_effect,
                    "collect_plot_data": collect_plot_data,
                    "onset_threshold": onset_threshold,
                    "onset_persistence": onset_persistence,
                },
                "scoring": {
                    "answer_score": "mean_t log P(y_t | target_prompt, y_<t, patch(S))",
                    "margin": "mean_i[q(y_source)-q(y_target)]",
                    "normalized_effect": "(m(S)-m(empty))/(m(all)-m(empty))",
                    "patch_position": "last token of prompt plus shared candidate prefix",
                },
                "evaluations": [],
                "model_inference_count": 0,
                "model_forward_count": 0,
            }
        )

    def publish_trace() -> None:
        if search_trace is not None and trace_callback is not None:
            trace_callback(search_trace)

    def raw_score(patch_layers: Sequence[int]) -> Tuple[Dict[str, Any], bool]:
        cache_key = tuple(patch_layers)
        cache_hit = cache_key in evaluation_cache
        if not cache_hit:
            site = config_to_site({"o": {"layers": list(patch_layers)}}, api=api, model=model)
            evaluation_cache[cache_key] = get_patched_full_answer_scores(
                nnmodel=nnmodel,
                processor=processor,
                patching_arguments=patching_arguments,
                source_answer_ids=source_answer_ids,
                target_answer_ids=target_answer_ids,
                site=site,
                batch_size=batch_size,
                max_samples=max_samples,
                max_length=max_length,
            )
            if search_trace is not None:
                search_trace["model_inference_count"] += 1
                search_trace["model_forward_count"] += evaluation_cache[cache_key]["model_forward_count"]
                search_trace["latest_completed_inference"] = {
                    "patch_layers": list(patch_layers),
                    "score": evaluation_cache[cache_key],
                }
                publish_trace()
        return evaluation_cache[cache_key], cache_hit

    baseline_score, baseline_cache_hit = raw_score([])
    full_score, full_cache_hit = raw_score(layer_indices)
    baseline_margin = float(baseline_score["margin"])
    full_margin = float(full_score["margin"])
    full_effect = full_margin - baseline_margin

    def normalized_effect(score: Dict[str, Any]) -> float:
        return (float(score["margin"]) - baseline_margin) / full_effect

    def record_evaluation(
        patch_layers: Sequence[int],
        phase: str,
        score: Dict[str, Any],
        cache_hit: bool,
        effect: Optional[float],
        **metadata: Any,
    ) -> Optional[int]:
        if search_trace is None:
            return None
        evaluation = {
            "evaluation_id": len(search_trace["evaluations"]),
            "phase": phase,
            "patch_layers": list(patch_layers),
            "cache_hit": cache_hit,
            "source_mean_logprob": float(score["source_mean_logprob"]),
            "target_mean_logprob": float(score["target_mean_logprob"]),
            "margin": float(score["margin"]),
            "normalized_effect": effect,
            "per_sample": {
                "source_mean_logprob": list(score["source_per_sample"]),
                "target_mean_logprob": list(score["target_per_sample"]),
                "margin": list(score["margin_per_sample"]),
                "source_token_count": list(score["source_token_counts"]),
                "target_token_count": list(score["target_token_counts"]),
            },
            "patched_forward_pairs": int(score["patched_forward_pairs"]),
            "model_forward_count": int(score["model_forward_count"]),
            "n_samples": int(score["n_samples"]),
        }
        evaluation.update(metadata)
        search_trace["evaluations"].append(evaluation)
        search_trace["logical_evaluation_count"] = len(search_trace["evaluations"])
        publish_trace()
        return evaluation["evaluation_id"]

    if search_trace is not None:
        search_trace["baseline_margin"] = baseline_margin
        search_trace["full_patch_margin"] = full_margin
        search_trace["full_patch_effect"] = full_effect

    if full_effect <= minimum_full_effect:
        record_evaluation([], "baseline_no_patch", baseline_score, baseline_cache_hit, None)
        record_evaluation(layer_indices, "reference_all_layers", full_score, full_cache_hit, None)
        if search_trace is not None:
            search_trace["status"] = "failed"
            search_trace["error"] = "Full-layer patch does not improve the source-vs-target complete-answer margin"
            publish_trace()
        raise RuntimeError(
            "Cannot normalize the layer-search effect: full-layer patch margin "
            f"({full_margin:.6f}) must exceed the no-patch margin "
            f"({baseline_margin:.6f}) by more than {minimum_full_effect:.2g}."
        )

    record_evaluation([], "baseline_no_patch", baseline_score, baseline_cache_hit, 0.0)
    record_evaluation(
        layer_indices,
        "reference_all_layers",
        full_score,
        full_cache_hit,
        1.0,
    )

    def evaluate_layers(
        patch_layers: Sequence[int],
        phase: str,
        **metadata: Any,
    ) -> Tuple[Dict[str, Any], float, Optional[int]]:
        score, cache_hit = raw_score(patch_layers)
        effect = normalized_effect(score)
        evaluation_id = record_evaluation(
            patch_layers,
            phase,
            score,
            cache_hit,
            effect,
            **metadata,
        )
        logger.info(
            "%s patch=%s: source=%.4f target=%.4f margin=%.4f R=%.4f",
            phase,
            list(patch_layers),
            score["source_mean_logprob"],
            score["target_mean_logprob"],
            score["margin"],
            effect,
        )
        return score, effect, evaluation_id

    logger.info(
        "Full-answer normalization: m(empty)=%.4f, m(all)=%.4f, delta=%.4f",
        baseline_margin,
        full_margin,
        full_effect,
    )

    supplied_upper_bound = upper_bound is not None
    if upper_bound is None:
        left, right = 1, n_search
        while left < right:
            candidate = (left + right) // 2
            _, effect, _ = evaluate_layers(
                layer_indices[:candidate],
                "upper_bound_search",
                candidate_position=candidate,
                candidate_layer=layer_indices[candidate - 1],
            )
            if effect >= retention_threshold:
                right = candidate
            else:
                left = candidate + 1
        upper_bound = left

    _, upper_effect, _ = evaluate_layers(
        layer_indices[:upper_bound],
        "provided_upper_bound" if supplied_upper_bound else "selected_upper_bound",
        candidate_position=upper_bound,
        candidate_layer=layer_indices[upper_bound - 1],
    )
    if upper_effect < retention_threshold:
        raise RuntimeError(
            f"Upper-bound prefix reaches R={upper_effect:.4f}, below the required {retention_threshold:.4f}."
        )

    if lower_bound is not None and lower_bound >= upper_bound:
        raise ValueError(f"lower_bound ({lower_bound}) must be smaller than upper_bound ({upper_bound})")

    if lower_bound is None:
        left, right = 0, upper_bound - 1
        best_lower_bound = 0
        while left <= right:
            candidate = (left + right) // 2
            _, effect, _ = evaluate_layers(
                layer_indices[candidate:upper_bound],
                "lower_bound_search",
                candidate_position=candidate,
                candidate_layer=layer_indices[candidate],
            )
            if effect >= retention_threshold:
                best_lower_bound = candidate
                left = candidate + 1
            else:
                right = candidate - 1
        lower_bound = best_lower_bound

    selected_layers = layer_indices[lower_bound:upper_bound]
    _, selected_effect, final_evaluation_id = evaluate_layers(
        selected_layers,
        "selected_range",
        lower_bound_position=lower_bound,
        upper_bound_position=upper_bound,
    )

    if selected_effect < retention_threshold:
        if search_trace is not None:
            search_trace["status"] = "failed"
            search_trace["error"] = "Selected range failed the retention threshold"
            publish_trace()
        raise RuntimeError(
            f"Selected range {selected_layers} reaches R={selected_effect:.4f}, "
            f"below the required {retention_threshold:.4f}."
        )

    if search_trace is not None:
        search_trace["status"] = "search_complete_plot_data_pending" if collect_plot_data else "complete"
        search_trace["selected_layers"] = list(selected_layers)
        search_trace["selected_normalized_effect"] = selected_effect
        search_trace["lower_bound_position"] = lower_bound
        search_trace["upper_bound_position"] = upper_bound
        search_trace["final_evaluation_id"] = final_evaluation_id
        search_trace["logical_evaluation_count"] = len(search_trace["evaluations"])
        search_trace.pop("latest_completed_inference", None)
        publish_trace()

    if collect_plot_data and search_trace is not None:
        search_trace["plot_data_status"] = "running"
        publish_trace()
        search_trace["probability_gap_plot"] = collect_patch_scope_gap_data(
            nnmodel=nnmodel,
            processor=processor,
            patching_arguments=patching_arguments,
            patch_layers=selected_layers,
            model=model,
            api=api,
            batch_size=batch_size,
            max_samples=max_samples,
        )
        onset = annotate_difference_onset(
            search_trace,
            threshold=onset_threshold,
            persistence=onset_persistence,
        )
        if onset["used_fallback"]:
            logger.info(
                "No sustained nondecreasing probability-gap onset inside the "
                "important range; retaining final important layer %d",
                onset["layer"],
            )
        else:
            logger.info(
                "Selected nondecreasing probability-gap onset layer %d "
                "(effect=%.2f, threshold=%.2f, peak=%.2f at layer %d)",
                onset["layer"],
                onset["effect_at_layer"],
                onset["threshold"],
                onset["peak_effect"],
                onset["peak_layer"],
            )
        search_trace["plot_data_inference_count"] = 2
        search_trace["plot_data_status"] = "complete"
        publish_trace()

    if search_trace is not None:
        search_trace["status"] = "complete"
        publish_trace()
    return selected_layers


def auto_search(
    model,
    processor,
    patching_arguments,
    n_layers: Optional[int] = None,
    eps: float = 0.3,
    upper_thres: float = 0.85,
    lower_thres: Optional[float] = None,
    phi: float = 0.05,
    batch_size: int = 16,
    cliff_z=3.5,
    api=Qwen2p5VL,
    max_samples: Optional[int] = None,
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
    search_trace: Optional[MutableMapping[str, Any]] = None,
) -> List[int]:
    """Auto search for the important layer range.

    Args:
        model: The model to analyze.
        processor: Processor for the model.
        patching_arguments: Patching data container with source/target tensors.
        n_layers: Number of layers in model (auto-detected if None).
        eps: Epsilon threshold for refinement step.
        upper_thres: Probability threshold for the upper bound search.
          Scaled by ``best_max_prob`` internally.
        lower_thres: Probability threshold for the lower bound search.
          Scaled by ``best_max_prob`` internally. When ``None``
          (default), falls back to ``upper_thres``.
        phi: Phi threshold for refinement candidate selection.
        batch_size: Batch size for inference.
        max_samples: Maximum number of samples to use.
        lower_bound: Optional starting position within ``layer_indices``.
        upper_bound: Optional exclusive ending position within ``layer_indices``.
        search_trace: Optional mutable mapping populated with every probability
          evaluation. Supplying it does not trigger additional inference.
        cliff_z:

    Returns:
        List of layer indices in the important range.
    """
    if n_layers is None:
        n_layers = get_num_layers(model)

    if upper_thres is None:
        raise ValueError("upper_thres must be a numeric threshold ratio")
    if not 0 <= upper_thres <= 1:
        raise ValueError(f"upper_thres must be in [0, 1], got {upper_thres}")
    if lower_thres is not None and not 0 <= lower_thres <= 1:
        raise ValueError(f"lower_thres must be in [0, 1], got {lower_thres}")
    if eps < 0 or phi < 0:
        raise ValueError("eps and phi must be non-negative")

    if layer_indices is None:
        layer_indices = list(range(n_layers))
    else:
        layer_indices = list(layer_indices)
    if len(layer_indices) < 2:
        raise ValueError("Layer search requires at least two searchable layers")
    if layer_indices != sorted(set(layer_indices)):
        raise ValueError("layer_indices must be strictly increasing and unique")
    if layer_indices[0] < 0 or layer_indices[-1] >= n_layers:
        raise ValueError(f"layer_indices must be within [0, {n_layers - 1}], got {layer_indices}")

    n_search = len(layer_indices)

    if upper_bound is not None and not 1 <= upper_bound <= n_search:
        raise ValueError(f"upper_bound is an exclusive position and must be in [1, {n_search}]")
    if lower_bound is not None and not 0 <= lower_bound < n_search:
        raise ValueError(f"lower_bound is a position and must be in [0, {n_search - 1}]")

    nnmodel = NNsight(model)

    # Common arguments for get_patch_scope_probs
    common_kwargs = dict(
        nnmodel=nnmodel,
        processor=processor,
        source_input_ids=patching_arguments.source_input_ids,
        target_input_ids=patching_arguments.target_input_ids,
        source_attention_mask=patching_arguments.source_attention_mask,
        target_attention_mask=patching_arguments.target_attention_mask,
        source_mm_token_type_ids=patching_arguments.source_mm_token_type_ids,
        target_mm_token_type_ids=patching_arguments.target_mm_token_type_ids,
        source_answer=patching_arguments.source_answer_index,
        target_answer=patching_arguments.target_answer_index,
        source_pixel_values=patching_arguments.source_pixel_values,
        target_pixel_values=patching_arguments.target_pixel_values,
        source_image_grid_thw=patching_arguments.source_image_grid_thw,
        target_image_grid_thw=patching_arguments.target_image_grid_thw,
        source_image_position_ids=patching_arguments.source_image_position_ids,
        target_image_position_ids=patching_arguments.target_image_position_ids,
        batch_size=batch_size,
        max_samples=max_samples,
    )

    if search_trace is not None:
        search_trace.clear()
        search_trace.update(
            {
                "schema_version": 1,
                "status": "running",
                "n_layers": n_layers,
                "searchable_layers": list(layer_indices),
                "parameters": {
                    "eps": eps,
                    "upper_threshold_ratio": upper_thres,
                    "lower_threshold_ratio": lower_thres,
                    "phi": phi,
                    "cliff_z": cliff_z,
                    "batch_size": batch_size,
                    "max_samples": max_samples,
                },
                "evaluations": [],
                "model_inference_count": 0,
            }
        )

    evaluation_cache = {}

    def evaluate_layers(
        patch_layers: List[int],
        phase: str,
        **metadata: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Evaluate one candidate and optionally retain the exact plot data."""
        cache_key = tuple(patch_layers)
        cache_hit = cache_key in evaluation_cache
        if not cache_hit:
            site_config = {"o": {"layers": list(patch_layers)}}
            site = config_to_site(site_config, api=api, model=model)
            evaluation_cache[cache_key] = get_patch_scope_probs(
                **common_kwargs,
                site=site,
                return_stats=search_trace is not None,
            )
            if search_trace is not None:
                search_trace["model_inference_count"] += 1

        if search_trace is None:
            return evaluation_cache[cache_key]

        source_stats, target_stats = evaluation_cache[cache_key]
        p_mean_source = source_stats[0]
        p_mean_target = target_stats[0]
        source_values = p_mean_source.detach().float().cpu().reshape(-1)
        target_values = p_mean_target.detach().float().cpu().reshape(-1)
        evaluation = {
            "evaluation_id": len(search_trace["evaluations"]),
            "phase": phase,
            "patch_layers": list(patch_layers),
            "cache_hit": cache_hit,
            "source": _probability_stats_to_trace(source_stats),
            "target": _probability_stats_to_trace(target_stats),
            "max_source_probability": float(source_values.max().item()),
            "output_source_probability": float(source_values[-1].item()),
            "output_target_probability": float(target_values[-1].item()),
            "output_margin": float(source_values[-1].item() - target_values[-1].item()),
        }
        evaluation.update(metadata)
        search_trace["evaluations"].append(evaluation)
        return p_mean_source, p_mean_target

    p_mean_source, _ = evaluate_layers(layer_indices, "initial_all_layers")
    best_max_prob = p_mean_source.max().item()
    logger.info(f"Initial full-layer patch - Max probability: {best_max_prob:.4f}")
    if lower_thres is None:
        lower_thres = upper_thres
    upper_thres_val = best_max_prob * upper_thres
    lower_thres_val = best_max_prob * lower_thres
    if search_trace is not None:
        search_trace["parameters"]["lower_threshold_ratio"] = lower_thres
        search_trace["best_max_source_probability"] = best_max_prob
        search_trace["upper_threshold_value"] = upper_thres_val
        search_trace["lower_threshold_value"] = lower_thres_val

    if upper_bound is None:
        logger.info(f"Step 1: Searching for upper bound (binary search, thres={upper_thres_val})...")
        left, right = 0, n_search
        while left < right:
            candidate = (left + right) // 2
            logger.info(
                "Trying upper-bound position %d (prefix ends before layer %d)...",
                candidate,
                layer_indices[candidate],
            )
            p_mean_source, _ = evaluate_layers(
                layer_indices[:candidate],
                "upper_bound_search",
                candidate_position=candidate,
                candidate_layer=layer_indices[candidate],
            )
            logger.info(
                "Upper-bound position %d - Max probability: %.4f, probability at readout layer %d: %.4f",
                candidate,
                p_mean_source.max().item(),
                layer_indices[candidate],
                p_mean_source[layer_indices[candidate], 0].item(),
            )
            if p_mean_source.max().item() >= upper_thres_val:
                right = candidate
            else:
                left = candidate + 1
        upper_bound = left
        if upper_bound <= 0:
            raise RuntimeError(
                "Layer search selected an empty upper prefix; check that source and "
                "target answer tokens differ and that the unpatched target does not "
                "already predict the source answer."
            )
        p_mean_source, _ = evaluate_layers(
            layer_indices[:upper_bound],
            "selected_upper_bound",
            candidate_position=upper_bound,
            candidate_layer=layer_indices[upper_bound - 1],
        )
        logger.info(
            f"Get Upper bound: {upper_bound} (layer {layer_indices[upper_bound - 1]}) - Max probability: {p_mean_source.max().item():.4f}, "
            f"Probability at layer {layer_indices[upper_bound - 1]}: {p_mean_source[layer_indices[upper_bound - 1], 0].item():.4f}"
        )
    else:
        logger.info("Step 1: Using provided upper bound.")
        # Confirm upper-bound probabilities
        p_mean_source, _ = evaluate_layers(
            layer_indices[:upper_bound],
            "provided_upper_bound",
            candidate_position=upper_bound,
            candidate_layer=layer_indices[upper_bound - 1],
        )
        logger.info(
            f"Get Upper bound: {upper_bound} (layer {layer_indices[upper_bound - 1]}) - Max probability: {p_mean_source.max().item():.4f}, "
            f"Probability at layer {layer_indices[upper_bound - 1]}: {p_mean_source[layer_indices[upper_bound - 1], 0].item():.4f}"
        )

    if lower_bound is not None and lower_bound >= upper_bound:
        raise ValueError(f"lower_bound ({lower_bound}) must be smaller than upper_bound ({upper_bound})")

    # --- Lower bound ---
    if lower_bound is None:
        logger.info(f"Step 2: Searching for lower bound (binary search, thres={lower_thres_val})...")
        left, right = 0, upper_bound - 1
        best_lower_bound = None
        lower_bound_history = {}
        while left <= right:
            candidate = (left + right) // 2
            p_mean_source, _ = evaluate_layers(
                layer_indices[candidate:upper_bound],
                "lower_bound_search",
                candidate_position=candidate,
                candidate_layer=layer_indices[candidate] if candidate < n_search else None,
            )
            max_p = p_mean_source.max().item()
            lower_bound_history[candidate] = max_p
            logger.info(
                f"Lower bound: {candidate} (layer {layer_indices[candidate]}) - Max probability: {max_p:.4f}, "
                f"Probability at layer {layer_indices[candidate]}: {p_mean_source[layer_indices[candidate], 0].item():.4f}, "
                f"Probability at last layer: {p_mean_source[-1, 0].item():.4f}"
            )
            if max_p >= lower_thres_val:
                best_lower_bound = candidate
                left = candidate + 1
            else:
                right = candidate - 1
        if best_lower_bound is None:
            raise RuntimeError(
                "No non-empty lower-to-upper layer range reaches the lower "
                f"probability threshold {lower_thres_val:.4f}."
            )
        lower_bound = best_lower_bound
        p_mean_source, _ = evaluate_layers(
            layer_indices[lower_bound:upper_bound],
            "selected_lower_bound",
            candidate_position=lower_bound,
            candidate_layer=layer_indices[lower_bound],
        )
        logger.info(
            f"Final Lower bound: {lower_bound} (layer {layer_indices[lower_bound]}) - Max probability: {p_mean_source.max().item():.4f}, "
            f"Probability at layer {layer_indices[lower_bound]}: {p_mean_source[layer_indices[lower_bound], 0].item():.4f}, "
            f"Probability at last layer: {p_mean_source[-1, 0].item():.4f}"
        )
        base_range = layer_indices[lower_bound:upper_bound]
    else:
        logger.info("Step 2: Using provided lower bound.")
        logger.info(
            f"Lower bound: {lower_bound} (layer {layer_indices[lower_bound]}) - Max probability: {p_mean_source.max().item():.4f}, "
            f"Probability at layer {layer_indices[lower_bound]}: {p_mean_source[layer_indices[lower_bound], 0].item():.4f}, "
            f"Probability at last layer: {p_mean_source[-1, 0].item():.4f}"
        )
        base_range = layer_indices[lower_bound:upper_bound]

    # Get probabilities for the final lower-to-upper range
    p_mean_source, p_mean_target = evaluate_layers(
        base_range,
        "selected_range",
        lower_bound_position=lower_bound,
        upper_bound_position=upper_bound,
    )
    logger.info(
        f"Lower bound: {lower_bound} - Upper bound: {upper_bound} - Max probability: {p_mean_source.max().item():.4f}, "
        f"Probability at last layer: {p_mean_source[-1, 0].item():.4f}"
    )

    # # ------------------------------------------------------------- #
    # # Step 3: Refinement
    # # ------------------------------------------------------------- #
    logger.info("Step 3: Refining...")
    # Refine the early important layers if found
    if "lower_bound_history" in locals() and lower_bound_history:
        # Check for huge differences across evaluated lower bounds
        sorted_cands = sorted(lower_bound_history.keys())
        pairs = list(zip(sorted_cands[:-1], sorted_cands[1:]))
        slopes = [
            (lower_bound_history[c1] - lower_bound_history[c2]) / (layer_indices[c2] - layer_indices[c1])
            for c1, c2 in pairs
        ]
        zscores = _modified_zscores(slopes)
        early_important_positions = []
        for (c1, c2), slope, z in zip(pairs, slopes, zscores):
            if z > cliff_z and slope > 0:
                logger.info(
                    f"Cliff detected between lower bounds {c1} (prob={lower_bound_history[c1]:.4f}) and {c2} (prob={lower_bound_history[c2]:.4f})"
                )
                early_important_positions.extend(list(range(c1, c2)))
        if early_important_positions:
            early_important_layers = [layer_indices[pos] for pos in early_important_positions]
            logger.info(f"Keeping early important layers: {early_important_layers}")
            logger.info("Redoing lower bound search for the remaining layers...")
            left = max(early_important_positions) + 1
            right = upper_bound
            best_new_lower_bound = None
            while left <= right:
                candidate = (left + right) // 2
                test_range = early_important_layers + layer_indices[candidate:upper_bound]
                p_mean_source, _ = evaluate_layers(
                    test_range,
                    "early_cliff_lower_bound_search",
                    candidate_position=candidate,
                    candidate_layer=layer_indices[candidate] if candidate < n_search else None,
                    early_important_layers=list(early_important_layers),
                )
                max_p = p_mean_source.max().item()
                logger.info(
                    "Redo lower-bound position %d (layer %s) - Max probability: %.4f",
                    candidate,
                    layer_indices[candidate] if candidate < n_search else "end",
                    max_p,
                )
                if max_p >= lower_thres_val:
                    best_new_lower_bound = candidate
                    left = candidate + 1
                else:
                    right = candidate - 1
            if best_new_lower_bound is None:
                raise RuntimeError(
                    "Early-cliff refinement found no range that reaches the "
                    f"lower probability threshold {lower_thres_val:.4f}."
                )
            new_lower_bound = best_new_lower_bound
            logger.info(
                "Redo final lower-bound position: %d (layer %s)",
                new_lower_bound,
                layer_indices[new_lower_bound] if new_lower_bound < n_search else "end",
            )
            lower_bound = new_lower_bound
            base_range = _combine_early_and_tail_layers(
                early_important_positions,
                layer_indices,
                lower_bound,
                upper_bound,
            )

            # Re-evaluate to get final p_mean_source for the loop condition
            p_mean_source, p_mean_target = evaluate_layers(
                base_range,
                "selected_range_after_early_cliff",
                lower_bound_position=lower_bound,
                upper_bound_position=upper_bound,
            )

    while p_mean_source[-1, 0].item() <= p_mean_target[-1, 0].item() + eps:
        logger.info(
            f"Refining (Probability source: {p_mean_source[-1, 0].item():.4f}, "
            f"Probability target: {p_mean_target[-1, 0].item():.4f}, "
            f"Difference: {p_mean_source[-1, 0].item() - p_mean_target[-1, 0].item()}) (eps={eps})..."
        )
        logger.info("Current base range: %s", base_range)

        candidate_layer = _find_refinement_candidate(
            p_mean_source,
            layer_indices,
            base_range,
            phi,
        )
        if candidate_layer is None:
            raise RuntimeError(
                "Layer-search refinement margin is still unmet, but no later "
                f"probability cliff exceeds phi={phi:.4f}. Refusing to return "
                "an unverified important-layer range."
            )
        base_range.append(candidate_layer)
        logger.info("Refined base range: %s", base_range)

        p_mean_source, p_mean_target = evaluate_layers(
            base_range,
            "tail_refinement",
            added_layer=candidate_layer,
        )
        logger.info(
            f"Out probability source: {p_mean_source[-1, 0].item():.4f}, "
            f"Out probability target: {p_mean_target[-1, 0].item():.4f}, "
            f"difference: {p_mean_source[-1, 0].item() - p_mean_target[-1, 0].item()} (eps={eps})"
        )

    if search_trace is not None:
        search_trace["status"] = "complete"
        search_trace["selected_layers"] = list(base_range)
        search_trace["lower_bound_position"] = lower_bound
        search_trace["upper_bound_position"] = upper_bound
        search_trace["final_evaluation_id"] = len(search_trace["evaluations"]) - 1
        search_trace["logical_evaluation_count"] = len(search_trace["evaluations"])
    return base_range


def filter_confident_samples(
    model,
    target_input_ids: Tensor,
    target_inverted_input_ids: Tensor,
    target_index: Tensor,
    source_index: Tensor,
    attention_mask_target: Tensor,
    attention_mask_target_inverted: Tensor,
    mm_token_type_ids_target: Optional[Tensor] = None,
    mm_token_type_ids_target_inverted: Optional[Tensor] = None,
    batch_size: int = 48,
) -> Tensor:
    confident_indices = []
    device = next(model.parameters()).device

    for i in trange(0, target_input_ids.shape[0], batch_size):
        batch_end = min(i + batch_size, target_input_ids.shape[0])

        batch_tt = target_input_ids[i:batch_end].to(device)
        batch_tit = target_inverted_input_ids[i:batch_end].to(device)
        batch_ti = target_index[i:batch_end].to(device)
        batch_si = source_index[i:batch_end].to(device)
        batch_amt = attention_mask_target[i:batch_end].to(device)
        batch_amti = attention_mask_target_inverted[i:batch_end].to(device)

        tt_kwargs = {"input_ids": batch_tt, "attention_mask": batch_amt}
        tit_kwargs = {"input_ids": batch_tit, "attention_mask": batch_amti}

        if mm_token_type_ids_target is not None:
            tt_kwargs["mm_token_type_ids"] = mm_token_type_ids_target[i:batch_end].to(device)
        if mm_token_type_ids_target_inverted is not None:
            tit_kwargs["mm_token_type_ids"] = mm_token_type_ids_target_inverted[i:batch_end].to(device)

        with torch.no_grad():
            outputs_tt = model(**tt_kwargs).logits
            outputs_tit = model(**tit_kwargs).logits

        max_logit_indices_tt = outputs_tt[:, -1, :].argmax(dim=-1)
        max_logit_indices_tit = outputs_tit[:, -1, :].argmax(dim=-1)

        match_tt = max_logit_indices_tt == batch_ti
        match_tit = max_logit_indices_tit == batch_si
        not_same_answer_in_gt = batch_ti != batch_si

        confident_batch = torch.where(match_tt & match_tit & not_same_answer_in_gt)[0] + i
        confident_indices.extend(confident_batch.tolist())

    confident_indices = torch.tensor(confident_indices)
    logger.info(f"Number of confident samples: {len(confident_indices)}/{target_input_ids.shape[0]}")
    return confident_indices


def inputs_collator(inputs: Dict, device: torch.device) -> Dict:
    for k, v in inputs.items():
        if "->" in k:
            inputs[k] = v.to(device)
        elif "subspace" in k:
            inputs[k] = v.tolist() if hasattr(v, "tolist") else v
        elif v is not None:
            if "pixel_values" in k:
                if "image_position_ids" not in inputs:
                    v = v.view(-1, v.size(-1))
            if "image_position_ids" in k:
                # image_position_ids should be passed as-is (not reshaped)
                pass
            inputs[k] = v.to(device)
    return inputs


class LowRankOrthogonalProjection(nn.Module, PyTorchModelHubMixin):
    """
    A projection that replaces the subspace (spanned by self.weight) value in the target activation with the one in the source activation.

    This is derived from the pyvene library and the implementation for https://arxiv.org/abs/2411.07404.
    """

    def __init__(self, embed_dim, rank=1, orthogonalize=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embed_dim, rank), requires_grad=True)
        torch.nn.init.orthogonal_(self.weight)

        self._P = None
        self._orthogonal_complement = None
        if orthogonalize:
            self = nn.utils.parametrizations.orthogonal(self)

    def project(self, x):
        """
        Computes the dot product of x and the subspace spanned by self.weight. This results in the value of x in the subspace spanned by self.weight.

        Args:
            x (torch.Tensor): The vector to project (batch_size, embed_dim)

        Returns:
            torch.Tensor: The value of x in the subspace spanned by self.weight (batch_size)
        """
        return torch.matmul(x.to(self.weight.dtype), self.weight)

    def get_P(self):
        """
        Returns the projection matrix P. This matrix will project any vector onto the subspace spanned by self.weight.
        """
        if self._P is None or self.training:
            self._P = torch.matmul(self.weight, self.weight.T)
        return self._P

    def get_orthogonal_complement(self, P=None):
        """
        Returns the projection matrix onto the orthogonal complement of the subspace spanned by self.weight.
        """
        # recompute P
        P = self.get_P()
        if self._orthogonal_complement is None or self.training:
            identity = torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
            self._orthogonal_complement = identity - P
        return self._orthogonal_complement

    def forward(self, source, target):
        """
        Replace the subspace (spanned by self.weight) value in the target activation with the one in the source activation.

        Args:
            source (torch.Tensor): The source activation (batch_size, seq_len, embed_dim)
            target (torch.Tensor): The target activation (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: The modified target activation where the subspace value is replaced by the one from the source activation.
        """
        P = self.get_P()
        orthogonal_complement = self.get_orthogonal_complement(P)
        # h_t = (I-P) h_t + P h_s
        return torch.matmul(target.to(self.weight.dtype), orthogonal_complement.T) + torch.matmul(
            source.to(self.weight.dtype), P.T
        )

    def constant_forward(self, source_constant, target):
        """
        Replace the subspace (spanned by self.weight) value in the target activation with a constant source value.
        If you are not training the subspace, make sure to call self.eval() before calling this function.
        This will ensure that the projection matrix P is not updated during the forward pass.

        Args:
            source_constant (torch.Tensor): The constant source value or steering value - called c(w) in the paper (batch_size)
            target (torch.Tensor): The target activation (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: The modified target activation where the subspace value is replaced by the constant source value.
        """
        P = self.get_P()
        orthogonal_complement = self.get_orthogonal_complement(P)
        # h_t = (I-P) h_t + P u c(w)
        return torch.matmul(target.to(self.weight.dtype), orthogonal_complement.T) + torch.matmul(
            source_constant.to(self.weight.dtype).unsqueeze(1), self.weight.T
        ).squeeze(1)

    def __str__(self):
        return "LowRankOrthogonalProjection()"

    @staticmethod
    def load_pretrained(path):
        warnings.warn(
            "Loading pretrained projection using load_pretrained() path is deprecated. Use the .from_pretrained() method instead."
        )
        state_dict = torch.load(path, weights_only=True)
        proj = LowRankOrthogonalProjection(state_dict["embed_dim"].item(), state_dict["rank"].item())
        state_dict.pop("embed_dim")
        state_dict.pop("rank")
        proj.load_state_dict(state_dict)
        return proj

    @staticmethod
    def from_subspace_basis(subspace_basis):
        """
        Create a projection from a subspace normal basis.

        Args:
            subspace_basis (torch.Tensor): The subspace basis (embed_dim, rank). Each column is a basis vector with unit norm.

        Returns:
            LowRankOrthogonalProjection: The projection that projects any vector onto the subspace spanned by subspace_basis.
        """
        assert len(subspace_basis.shape) == 2, "Subspace basis must be a 2D tensor of shape (embed_dim, rank)"
        rank = subspace_basis.shape[1]
        embed_dim = subspace_basis.shape[0]
        for basis in range(rank):
            assert subspace_basis[:, basis].norm() == 1, "Subspace basis must be normalized"
        proj = LowRankOrthogonalProjection(embed_dim, rank, orthogonalize=False)
        proj.weight = nn.Parameter(subspace_basis, requires_grad=True)
        proj = nn.utils.parametrizations.orthogonal(proj)
        return proj
