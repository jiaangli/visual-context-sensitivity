import logging
import os
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import AutoProcessor

from src.sft.data_process import build_sample, load_jsonl, resolve_jsonl_image_root
from src.sft.preference_eval import build_probability_inputs, sequence_logprobs

logger = logging.getLogger(__name__)

DEFAULT_DAS_TRAIN_PAIRS = 1024

_img_transform = v2.Compose(
    [
        v2.Resize((512, 512)),
        v2.CenterCrop(512),
    ]
)


@dataclass
class PatchingData:
    source_input_ids: Tensor
    target_input_ids: Tensor
    source_answer_index: Tensor
    target_answer_index: Tensor
    source_attention_mask: Tensor
    target_attention_mask: Tensor
    source_pixel_values: Optional[Tensor] = None
    target_pixel_values: Optional[Tensor] = None
    source_image_grid_thw: Optional[Tensor] = None
    target_image_grid_thw: Optional[Tensor] = None
    source_image_position_ids: Optional[Tensor] = None
    target_image_position_ids: Optional[Tensor] = None
    source_mm_token_type_ids: Optional[Tensor] = None
    target_mm_token_type_ids: Optional[Tensor] = None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({f: getattr(self, f) for f in self.__dataclass_fields__}, path)
        logger.info(f"Saved PatchingData to {path}")

    @classmethod
    def load(cls, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**state)


def prepare_vlm_patching_data(
    processor: AutoProcessor,
    dataset_path: str,
    image_root: Optional[str] = None,
    device: str = "cuda",
    use_image: bool = True,
) -> Tuple[List, List, PatchingData]:
    entries = load_jsonl(dataset_path)
    image_root = resolve_jsonl_image_root(dataset_path, entries, image_root)
    random.shuffle(entries)
    max_length = 512

    dataset_name = Path(dataset_path).stem
    samples = []
    context_samples = []  # for pixel_values
    context_metadatas = []
    prior_metadatas = []

    for entry in entries:
        ctx_sample = build_sample(
            entry,
            "context",
            dataset_name,
            image_root=image_root,
            use_image=use_image,
        )
        prior_sample = build_sample(
            entry,
            "prior",
            dataset_name,
            image_root=image_root,
            use_image=use_image,
        )

        samples.append(ctx_sample.messages)
        samples.append(prior_sample.messages)
        if use_image:
            context_samples.append(ctx_sample.messages)
        context_metadatas.append(ctx_sample.metadata)
        prior_metadatas.append(prior_sample.metadata)

    model_name = processor.tokenizer.name_or_path.lower()
    is_gemma4 = "gemma" in model_name

    if "qwen2" in model_name:
        input_data = processor.apply_chat_template(
            samples,
            processor_kwargs={
                "truncation": False,
                "padding": True,
            },
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
    elif is_gemma4:
        input_data = processor.apply_chat_template(
            samples,
            processor_kwargs={
                "truncation": False,
                "padding": True,
            },
            add_generation_prompt=False,
            enable_thinking=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        input_data = processor.apply_chat_template(
            samples,
            processor_kwargs={
                "truncation": False,
                "padding": True,
            },
            add_generation_prompt=False,
            enable_thinking=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

    if use_image and len(context_samples) > 0:
        if "qwen2" in model_name:
            pixel_values = processor.apply_chat_template(
                context_samples,
                processor_kwargs={
                    "truncation": False,
                    "padding": True,
                },
                add_generation_prompt=False,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )["pixel_values"]
        elif is_gemma4:
            pixel_values = processor.apply_chat_template(
                context_samples,
                processor_kwargs={
                    "truncation": False,
                    "padding": True,
                },
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            )["pixel_values"]
        else:
            pixel_values = processor.apply_chat_template(
                context_samples,
                processor_kwargs={
                    "truncation": False,
                    "padding": True,
                },
                add_generation_prompt=False,
                enable_thinking=False,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )["pixel_values"]

        if is_gemma4:
            # Gemma4 uses image_position_ids instead of image_grid_thw
            prior_encoded = {k: v[1::2] for k, v in input_data.items() if k != "pixel_values"}
            context_encoded = {k: v[0::2] for k, v in input_data.items() if k != "pixel_values"}
            # For Gemma4, pixel_values shape is already (batch, num_patches, dim)
            prior_encoded["pixel_values"] = pixel_values
            context_encoded["pixel_values"] = pixel_values
        else:
            patch_dim = pixel_values.size(1)
            pixel_values = pixel_values.view(len(context_samples), -1, patch_dim)
            assert pixel_values.size(1) == torch.prod(input_data["image_grid_thw"][0]).item()

            prior_encoded = {k: v[1::2] for k, v in input_data.items() if k != "pixel_values"}
            context_encoded = {k: v[0::2] for k, v in input_data.items() if k != "pixel_values"}
            prior_encoded["pixel_values"] = pixel_values
            context_encoded["pixel_values"] = pixel_values
    else:
        prior_encoded = {k: v[1::2] for k, v in input_data.items()}
        context_encoded = {k: v[0::2] for k, v in input_data.items()}

    context_answers = [e.context_answer for e in context_metadatas]
    prior_answers = [e.prior_answer for e in prior_metadatas]
    context_answer_ids = _encode_answers(processor.tokenizer, context_answers, device)
    prior_answer_ids = _encode_answers(processor.tokenizer, prior_answers, device)

    # Determine suffix tokens to trim (Gemma4 may differ)
    trim_suffix = 2

    # Handle image_position_ids for Gemma4
    source_image_position_ids = prior_encoded.get("image_position_ids")
    target_image_position_ids = context_encoded.get("image_position_ids")

    return (
        context_metadatas,
        prior_metadatas,
        PatchingData(
            source_input_ids=prior_encoded["input_ids"][:, :-trim_suffix],
            target_input_ids=context_encoded["input_ids"][:, :-trim_suffix],
            source_answer_index=prior_answer_ids,
            target_answer_index=context_answer_ids,
            source_attention_mask=prior_encoded["attention_mask"][:, :-trim_suffix],
            target_attention_mask=context_encoded["attention_mask"][:, :-trim_suffix],
            source_pixel_values=prior_encoded.get("pixel_values"),
            target_pixel_values=context_encoded.get("pixel_values"),
            source_image_grid_thw=prior_encoded.get("image_grid_thw"),
            target_image_grid_thw=context_encoded.get("image_grid_thw"),
            source_image_position_ids=source_image_position_ids,
            target_image_position_ids=target_image_position_ids,
            source_mm_token_type_ids=prior_encoded.get(
                "mm_token_type_ids", torch.zeros_like(prior_encoded["input_ids"])
            )[:, :-trim_suffix],
            target_mm_token_type_ids=context_encoded.get(
                "mm_token_type_ids", torch.zeros_like(context_encoded["input_ids"])
            )[:, :-trim_suffix],
        ),
    )


def _encode_answers(tokenizer, answers: List[str], device: str) -> Tensor:
    answer_ids = []
    for answer in answers:
        tokens = tokenizer.encode(answer, add_special_tokens=False)
        answer_ids.append(tokens[0] if tokens else tokenizer.unk_token_id)
    return torch.tensor(answer_ids)


def transform_patching_data(
    data: PatchingData,
    context_metadatas,
    prior_metadatas,
    task_direction: str,
    return_pair_indices: bool = False,
) -> PatchingData:
    def finish(result, source_indices, target_indices):
        if return_pair_indices:
            return result, list(source_indices), list(target_indices)
        else:
            return result

    identity_indices = list(range(data.source_input_ids.shape[0]))

    match task_direction:
        case "prior_to_context":
            return finish(data, identity_indices, identity_indices)

        case "context_to_prior":
            result = PatchingData(
                source_input_ids=data.target_input_ids,
                target_input_ids=data.source_input_ids,
                source_answer_index=data.target_answer_index,
                target_answer_index=data.source_answer_index,
                source_attention_mask=data.target_attention_mask,
                target_attention_mask=data.source_attention_mask,
            )
            if data.source_pixel_values is not None:
                result.source_pixel_values = data.target_pixel_values
                result.target_pixel_values = data.source_pixel_values
            if data.source_image_grid_thw is not None:
                result.source_image_grid_thw = data.target_image_grid_thw
                result.target_image_grid_thw = data.source_image_grid_thw
            if data.source_image_position_ids is not None:
                result.source_image_position_ids = data.target_image_position_ids
                result.target_image_position_ids = data.source_image_position_ids
            if data.source_mm_token_type_ids is not None:
                result.source_mm_token_type_ids = data.target_mm_token_type_ids
                result.target_mm_token_type_ids = data.source_mm_token_type_ids
            return finish(result, identity_indices, identity_indices)

        case "prior_only":
            n = data.source_input_ids.shape[0]
            if n < 2:
                raise ValueError("Need at least 2 samples for 'prior_only' mode")

            valid_pairs = [
                (i, j)
                for i in range(n)
                for j in range(n)
                if i < j
                and prior_metadatas[i].question != prior_metadatas[j].question
                and prior_metadatas[i].prior_answer != prior_metadatas[j].prior_answer
            ]
            if len(valid_pairs) < n:
                raise ValueError(f"Not enough valid pairs: {len(valid_pairs)} < {n}")

            random.shuffle(valid_pairs)

            src_idx = []
            tgt_idx = []
            used = set()
            for i, j in valid_pairs:
                if (i, j) not in used and (j, i) not in used:
                    src_idx.append(i)
                    tgt_idx.append(j)
                    used.add((i, j))
                    used.add((j, i))
                    if len(src_idx) == n:
                        break

            if len(src_idx) < n:
                raise ValueError(f"Could only find {len(src_idx)}/{n} unique pairs")

            src_idx = torch.tensor(src_idx, dtype=torch.long)
            tgt_idx = torch.tensor(tgt_idx, dtype=torch.long)

            result = PatchingData(
                source_input_ids=data.source_input_ids[src_idx],
                target_input_ids=data.source_input_ids[tgt_idx],
                source_answer_index=data.source_answer_index[src_idx],
                target_answer_index=data.source_answer_index[tgt_idx],
                source_attention_mask=data.source_attention_mask[src_idx],
                target_attention_mask=data.source_attention_mask[tgt_idx],
            )
            if data.source_pixel_values is not None:
                result.source_pixel_values = data.source_pixel_values[src_idx]
                result.target_pixel_values = data.source_pixel_values[tgt_idx]
            if data.source_image_grid_thw is not None:
                result.source_image_grid_thw = data.source_image_grid_thw[src_idx]
                result.target_image_grid_thw = data.source_image_grid_thw[tgt_idx]
            if data.source_image_position_ids is not None:
                result.source_image_position_ids = data.source_image_position_ids[src_idx]
                result.target_image_position_ids = data.source_image_position_ids[tgt_idx]
            if data.source_mm_token_type_ids is not None:
                result.source_mm_token_type_ids = data.source_mm_token_type_ids[src_idx]
                result.target_mm_token_type_ids = data.source_mm_token_type_ids[tgt_idx]
            return finish(result, src_idx.tolist(), tgt_idx.tolist())

        case "context_only":
            n = data.target_input_ids.shape[0]
            if n < 2:
                raise ValueError("Need at least 2 samples for 'context_only' mode")

            valid_pairs = [
                (i, j)
                for i in range(n)
                for j in range(n)
                if i < j
                and context_metadatas[i].question != context_metadatas[j].question
                and context_metadatas[i].context_answer != context_metadatas[j].context_answer
            ]
            if len(valid_pairs) < n:
                raise ValueError(f"Not enough valid pairs: {len(valid_pairs)} < {n}")

            random.shuffle(valid_pairs)

            src_idx = []
            tgt_idx = []
            used = set()
            for i, j in valid_pairs:
                if (i, j) not in used and (j, i) not in used:
                    src_idx.append(i)
                    tgt_idx.append(j)
                    used.add((i, j))
                    used.add((j, i))
                    if len(src_idx) == n:
                        break

            if len(src_idx) < n:
                raise ValueError(f"Could only find {len(src_idx)}/{n} unique pairs")

            src_idx = torch.tensor(src_idx, dtype=torch.long)
            tgt_idx = torch.tensor(tgt_idx, dtype=torch.long)

            result = PatchingData(
                source_input_ids=data.target_input_ids[src_idx],
                target_input_ids=data.target_input_ids[tgt_idx],
                source_answer_index=data.target_answer_index[src_idx],
                target_answer_index=data.target_answer_index[tgt_idx],
                source_attention_mask=data.target_attention_mask[src_idx],
                target_attention_mask=data.target_attention_mask[tgt_idx],
            )
            if data.target_pixel_values is not None:
                result.source_pixel_values = data.target_pixel_values[src_idx]
                result.target_pixel_values = data.target_pixel_values[tgt_idx]
            if data.target_image_grid_thw is not None:
                result.source_image_grid_thw = data.target_image_grid_thw[src_idx]
                result.target_image_grid_thw = data.target_image_grid_thw[tgt_idx]
            if data.target_image_position_ids is not None:
                result.source_image_position_ids = data.target_image_position_ids[src_idx]
                result.target_image_position_ids = data.target_image_position_ids[tgt_idx]
            if data.target_mm_token_type_ids is not None:
                result.source_mm_token_type_ids = data.target_mm_token_type_ids[src_idx]
                result.target_mm_token_type_ids = data.target_mm_token_type_ids[tgt_idx]
            return finish(result, src_idx.tolist(), tgt_idx.tolist())

        case _:
            raise ValueError(f"Unknown task_direction: {task_direction}")


def _compute_correct_indices(model, data: PatchingData, batch_size: int = 16, processor=None) -> List[int]:

    device = next(model.parameters()).device
    correct_indices = []

    patch_dim = None
    if data.source_pixel_values is not None:
        patch_dim = data.source_pixel_values.size(-1)

    for i in range(0, data.source_input_ids.shape[0], batch_size):
        batch_end = min(i + batch_size, data.source_input_ids.shape[0])

        source_batch = data.source_input_ids[i:batch_end].to(device)
        target_batch = data.target_input_ids[i:batch_end].to(device)
        source_mask = data.source_attention_mask[i:batch_end].to(device)
        target_mask = data.target_attention_mask[i:batch_end].to(device)
        source_ans = data.source_answer_index[i:batch_end].to(device)
        target_ans = data.target_answer_index[i:batch_end].to(device)

        source_pv = None
        target_pv = None
        source_grid = None
        target_grid = None
        source_img_pos_ids = None
        target_img_pos_ids = None
        if data.source_pixel_values is not None:
            source_pv = data.source_pixel_values[i:batch_end].to(device)
            target_pv = data.target_pixel_values[i:batch_end].to(device)
            if data.source_image_grid_thw is not None:
                source_grid = data.source_image_grid_thw[i:batch_end].to(device)
                target_grid = data.target_image_grid_thw[i:batch_end].to(device)
            if data.source_image_position_ids is not None:
                source_img_pos_ids = data.source_image_position_ids[i:batch_end].to(device)
                target_img_pos_ids = data.target_image_position_ids[i:batch_end].to(device)

        source_mm_ids = None
        target_mm_ids = None
        if data.source_mm_token_type_ids is not None:
            source_mm_ids = data.source_mm_token_type_ids[i:batch_end].to(device)
            target_mm_ids = data.target_mm_token_type_ids[i:batch_end].to(device)

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                source_kwargs = {"input_ids": source_batch, "attention_mask": source_mask}
                target_kwargs = {"input_ids": target_batch, "attention_mask": target_mask}

                if source_pv is not None and patch_dim is not None:
                    if source_img_pos_ids is not None:
                        source_kwargs["pixel_values"] = source_pv
                        target_kwargs["pixel_values"] = target_pv
                    else:
                        source_kwargs["pixel_values"] = source_pv.view(-1, patch_dim)
                        target_kwargs["pixel_values"] = target_pv.view(-1, patch_dim)
                    if source_grid is not None:
                        source_kwargs["image_grid_thw"] = source_grid
                        target_kwargs["image_grid_thw"] = target_grid
                    if source_img_pos_ids is not None:
                        source_kwargs["image_position_ids"] = source_img_pos_ids
                        target_kwargs["image_position_ids"] = target_img_pos_ids
                if source_mm_ids is not None:
                    source_kwargs["mm_token_type_ids"] = source_mm_ids
                    target_kwargs["mm_token_type_ids"] = target_mm_ids

                source_logits = model(**source_kwargs).logits
                target_logits = model(**target_kwargs).logits

        source_preds = source_logits[:, -1, :].argmax(dim=-1)
        target_preds = target_logits[:, -1, :].argmax(dim=-1)

        source_correct = source_preds == source_ans
        target_correct = target_preds == target_ans
        both_correct = source_correct & target_correct

        batch_correct = torch.where(both_correct)[0] + i
        correct_indices.extend(batch_correct.tolist())
    return correct_indices


def _side_inputs(
    data: PatchingData,
    side: str,
    start: int,
    end: int,
) -> dict[str, Tensor]:
    if side not in {"source", "target"}:
        raise ValueError(f"side must be 'source' or 'target', got {side!r}")
    inputs: dict[str, Tensor] = {
        "input_ids": getattr(data, f"{side}_input_ids")[start:end],
        "attention_mask": getattr(data, f"{side}_attention_mask")[start:end],
    }
    for name in (
        "pixel_values",
        "image_grid_thw",
        "image_position_ids",
        "mm_token_type_ids",
    ):
        value = getattr(data, f"{side}_{name}")
        if value is not None:
            inputs[name] = value[start:end]
    return inputs


def _move_probability_inputs(
    inputs: Mapping[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    moved = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    if "pixel_values" in moved and "image_position_ids" not in moved:
        pixel_values = moved["pixel_values"]
        if pixel_values.dim() == 3:
            moved["pixel_values"] = pixel_values.view(-1, pixel_values.size(-1))
    return moved


def _candidate_mean_logprobs(
    model,
    processor,
    data: PatchingData,
    side: str,
    start: int,
    end: int,
    candidate_ids: Sequence[Sequence[int]],
    *,
    max_length: int,
) -> Tensor:
    prompt_inputs = _side_inputs(data, side, start, end)
    model_inputs, labels = build_probability_inputs(
        prompt_inputs,
        candidate_ids,
        pad_token_id=processor.tokenizer.pad_token_id,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    model_inputs = _move_probability_inputs(model_inputs, device)
    labels = labels.to(device, non_blocking=True)
    autocast_enabled = device.type == "cuda"
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ),
    ):
        logits = model(**model_inputs).logits
        _, means, _ = sequence_logprobs(logits, labels)
    return means.detach().cpu()


def _select_patching_rows(data: PatchingData, indices: Sequence[int]) -> PatchingData:
    index = torch.tensor(indices, dtype=torch.long)
    selected: dict[str, Optional[Tensor]] = {}
    for name in data.__dataclass_fields__:
        value = getattr(data, name)
        selected[name] = None if value is None else value[index.to(value.device)]
    return PatchingData(**selected)


def filter_by_full_answer_preferences(
    model,
    processor,
    data: PatchingData,
    source_answer_ids: Sequence[Sequence[int]],
    target_answer_ids: Sequence[Sequence[int]],
    *,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    max_length: int = 512,
    cache_path=None,
    normalize_to_default_size: bool = True,
    require_distinct_answer_tokens: bool = True,
    return_filter_stats: bool = False,
):
    """Keep pairs correct under SFT's full-answer preference criterion.

    A pair is retained only when each prompt assigns a higher mean
    full-answer log-probability to its own complete answer than to the
    alternative answer. ``require_distinct_answer_tokens`` additionally keeps
    only pairs whose two complete answers begin with different token ids.
    """
    n_pairs = data.source_input_ids.shape[0]
    if len(source_answer_ids) != n_pairs or len(target_answer_ids) != n_pairs:
        raise ValueError("Full-answer token lists must align with PatchingData rows")

    if cache_path is not None and os.path.exists(cache_path):
        logger.info("Loading full-answer correct indices from cache: %s", cache_path)
        correct_indices = torch.load(cache_path, map_location="cpu", weights_only=True).tolist()
    else:
        correct_indices: list[int] = []
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            source_candidates = source_answer_ids[start:end]
            target_candidates = target_answer_ids[start:end]

            source_correct_scores = _candidate_mean_logprobs(
                model,
                processor,
                data,
                "source",
                start,
                end,
                source_candidates,
                max_length=max_length,
            )
            source_alternative_scores = _candidate_mean_logprobs(
                model,
                processor,
                data,
                "source",
                start,
                end,
                target_candidates,
                max_length=max_length,
            )
            target_correct_scores = _candidate_mean_logprobs(
                model,
                processor,
                data,
                "target",
                start,
                end,
                target_candidates,
                max_length=max_length,
            )
            target_alternative_scores = _candidate_mean_logprobs(
                model,
                processor,
                data,
                "target",
                start,
                end,
                source_candidates,
                max_length=max_length,
            )
            both_correct = (source_correct_scores > source_alternative_scores) & (
                target_correct_scores > target_alternative_scores
            )
            correct_indices.extend((torch.where(both_correct)[0] + start).tolist())

        logger.info(
            "%d/%d samples where full-answer preference is correct on both prompts",
            len(correct_indices),
            n_pairs,
        )
        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor(correct_indices, dtype=torch.long), cache_path)
            logger.info("Saved full-answer correct indices to cache: %s", cache_path)

    preference_correct_count = len(correct_indices)
    distinct_indices = {
        index
        for index, (source_ids, target_ids) in enumerate(zip(source_answer_ids, target_answer_ids))
        if source_ids and target_ids and source_ids[0] != target_ids[0]
    }
    if require_distinct_answer_tokens:
        correct_indices = [index for index in correct_indices if index in distinct_indices]
        logger.info(
            "%d/%d full-answer-preferred pairs have distinct first tokens",
            len(correct_indices),
            preference_correct_count,
        )

    eligible_count = len(correct_indices)
    random.shuffle(correct_indices)
    if max_samples is not None:
        correct_indices = correct_indices[:max_samples]
    elif normalize_to_default_size and len(correct_indices) < DEFAULT_DAS_TRAIN_PAIRS:
        if not correct_indices:
            raise ValueError("No correct full-answer preferences found")
        correct_indices += random.choices(
            correct_indices,
            k=DEFAULT_DAS_TRAIN_PAIRS - len(correct_indices),
        )
    elif normalize_to_default_size:
        correct_indices = correct_indices[:DEFAULT_DAS_TRAIN_PAIRS]

    if not correct_indices:
        raise ValueError("No correct full-answer preferences found")
    logger.info("%d full-answer samples to train", len(correct_indices))
    selected_data = _select_patching_rows(data, correct_indices)
    selected_source_answers = [list(source_answer_ids[index]) for index in correct_indices]
    selected_target_answers = [list(target_answer_ids[index]) for index in correct_indices]
    result = (
        selected_data,
        selected_source_answers,
        selected_target_answers,
        {
            "criterion": "full_answer_both_preferred",
            "total_training_pairs": n_pairs,
            "full_answer_both_preferred_pairs": preference_correct_count,
            "different_first_token_pairs": len(distinct_indices),
            "eligible_intersection_pairs": eligible_count,
            "selected_training_pairs": len(correct_indices),
        }
        if return_filter_stats
        else preference_correct_count,
    )
    return result


def filter_by_correct_predictions(
    model,
    data: PatchingData,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    processor=None,
    cache_path=None,
    require_distinct_answer_tokens: bool = False,
    require_distinct_answer_initials: bool = False,
    source_answer_texts: Optional[List[str]] = None,
    target_answer_texts: Optional[List[str]] = None,
    return_filter_stats: bool = False,
    return_selected_indices: bool = False,
    normalize_to_default_size: bool = True,
):
    if cache_path is not None and os.path.exists(cache_path):
        logger.info(f"Loading correct indices from cache: {cache_path}")
        correct_indices = torch.load(cache_path).tolist()
    else:
        correct_indices = _compute_correct_indices(model, data, batch_size=batch_size, processor=processor)
        logger.info(
            f"Computed correct indices: {len(correct_indices)}/{data.source_input_ids.shape[0]} samples where model is correct on both"
        )
        if cache_path is not None:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            torch.save(torch.tensor(correct_indices), cache_path)
            logger.info(f"Saved correct indices to cache: {cache_path}")

    argmax_correct_count = len(correct_indices)
    distinct_indices = set(torch.where(data.source_answer_index != data.target_answer_index)[0].cpu().tolist())
    if require_distinct_answer_tokens:
        correct_indices = [index for index in correct_indices if index in distinct_indices]
        logger.info(
            "%d/%d first-token-correct pairs have distinct answer first tokens",
            len(correct_indices),
            argmax_correct_count,
        )

    distinct_initial_indices: set[int] | None = None
    if require_distinct_answer_initials:
        if source_answer_texts is None or target_answer_texts is None:
            raise ValueError(
                "source_answer_texts and target_answer_texts are required when require_distinct_answer_initials=True"
            )
        if (
            len(source_answer_texts) != data.source_input_ids.shape[0]
            or len(target_answer_texts) != data.source_input_ids.shape[0]
        ):
            raise ValueError("Answer text lists must align with PatchingData rows")

        def initial(answer: str) -> str:
            normalized = unicodedata.normalize("NFKC", str(answer)).strip().casefold()
            if not normalized:
                raise ValueError("Answer text must contain a non-whitespace initial")
            return normalized[0]

        distinct_initial_indices = {
            index
            for index, (source_answer, target_answer) in enumerate(zip(source_answer_texts, target_answer_texts))
            if initial(source_answer) and initial(target_answer) and initial(source_answer) != initial(target_answer)
        }
        correct_indices = [index for index in correct_indices if index in distinct_initial_indices]
        logger.info(
            "%d/%d first-token-correct pairs have distinct answer initials",
            len(correct_indices),
            argmax_correct_count,
        )

    eligible_count = len(correct_indices)
    random.shuffle(correct_indices)
    if max_samples is not None:
        correct_indices = correct_indices[:max_samples]
    elif normalize_to_default_size and len(correct_indices) < DEFAULT_DAS_TRAIN_PAIRS:
        if len(correct_indices) == 0:
            raise ValueError("No correct predictions found")
        correct_indices += random.choices(
            correct_indices,
            k=DEFAULT_DAS_TRAIN_PAIRS - len(correct_indices),
        )

    elif normalize_to_default_size:
        correct_indices = correct_indices[:DEFAULT_DAS_TRAIN_PAIRS]

    idx = torch.tensor(correct_indices, dtype=torch.long)
    logger.info(f"Filtered to {len(idx)}/{data.source_input_ids.shape[0]} samples where model is correct on both")

    result = PatchingData(
        source_input_ids=data.source_input_ids[idx].to(model.device),
        target_input_ids=data.target_input_ids[idx].to(model.device),
        source_answer_index=data.source_answer_index[idx].to(model.device),
        target_answer_index=data.target_answer_index[idx].to(model.device),
        source_attention_mask=data.source_attention_mask[idx].to(model.device),
        target_attention_mask=data.target_attention_mask[idx].to(model.device),
    )
    if data.source_pixel_values is not None:
        result.source_pixel_values = data.source_pixel_values[idx].to(model.device)
    if data.target_pixel_values is not None:
        result.target_pixel_values = data.target_pixel_values[idx].to(model.device)
    if data.source_image_grid_thw is not None:
        result.source_image_grid_thw = data.source_image_grid_thw[idx].to(model.device)
    if data.target_image_grid_thw is not None:
        result.target_image_grid_thw = data.target_image_grid_thw[idx].to(model.device)
    if data.source_image_position_ids is not None:
        result.source_image_position_ids = data.source_image_position_ids[idx].to(model.device)
    if data.target_image_position_ids is not None:
        result.target_image_position_ids = data.target_image_position_ids[idx].to(model.device)
    if data.source_mm_token_type_ids is not None:
        result.source_mm_token_type_ids = data.source_mm_token_type_ids[idx].to(model.device)
    if data.target_mm_token_type_ids is not None:
        result.target_mm_token_type_ids = data.target_mm_token_type_ids[idx].to(model.device)

    if return_filter_stats:
        stats = {
            "criterion": "first_token_argmax_both_correct",
            "total_training_pairs": int(data.source_input_ids.shape[0]),
            "argmax_both_correct_pairs": argmax_correct_count,
            "different_first_token_pairs": len(distinct_indices),
            "different_answer_initial_pairs": len(distinct_initial_indices)
            if distinct_initial_indices is not None
            else None,
            "eligible_intersection_pairs": eligible_count,
            "selected_training_pairs": len(correct_indices),
        }
        if return_selected_indices:
            return result, stats, list(correct_indices)
        return result, stats
    if return_selected_indices:
        return result, list(correct_indices)
    return result


def expand_patching_data(
    data: PatchingData,
    indices=None,
) -> PatchingData:
    """
    Expand source and target fields into interleaved format.

    Given N pairs (source[i], target[i]), produces 2N samples:
    - source_expanded: [source[0], target[0], source[1], target[1], ...]
    - target_expanded: [target[0], source[0], target[1], source[1], ...]

    Args:
        data: PatchingData with source_*/target_* fields of shape (N, ...)
        indices: Optional tensor of indices to select. If None, uses all samples.

    Returns:
        PatchingData with expanded fields of shape (2N, ...)
    """
    if indices is not None:
        idx = indices.detach().cpu()
    else:
        idx = None

    # Fields to expand (source_X and target_X pairs)
    field_pairs = [
        ("source_input_ids", "target_input_ids"),
        ("source_answer_index", "target_answer_index"),
        ("source_attention_mask", "target_attention_mask"),
        ("source_pixel_values", "target_pixel_values"),
        ("source_image_grid_thw", "target_image_grid_thw"),
        ("source_image_position_ids", "target_image_position_ids"),
        ("source_mm_token_type_ids", "target_mm_token_type_ids"),
    ]

    expanded_data = {}

    for source_field, target_field in field_pairs:
        source_tensor = getattr(data, source_field)
        target_tensor = getattr(data, target_field)

        if source_tensor is None or target_tensor is None:
            continue

        if idx is not None:
            source_tensor = source_tensor[idx]
            target_tensor = target_tensor[idx]

        source_expanded = torch.stack([source_tensor, target_tensor], dim=1).view(-1, *source_tensor.shape[1:])
        target_expanded = torch.stack([target_tensor, source_tensor], dim=1).view(-1, *target_tensor.shape[1:])

        expanded_data[source_field] = source_expanded
        expanded_data[target_field] = target_expanded

    return PatchingData(**expanded_data)


class DASPairDataset(Dataset):
    def __init__(self, data: PatchingData, indices: Tensor):
        idx = indices.cpu()

        self.source_input_ids = data.source_input_ids[idx].cpu()
        self.target_input_ids = data.target_input_ids[idx].cpu()
        self.source_answer_index = data.source_answer_index[idx].cpu()
        self.target_answer_index = data.target_answer_index[idx].cpu()
        self.source_attention_mask = data.source_attention_mask[idx].cpu()
        self.target_attention_mask = data.target_attention_mask[idx].cpu()

        self.source_pixel_values = data.source_pixel_values[idx].cpu() if data.source_pixel_values is not None else None
        self.target_pixel_values = data.target_pixel_values[idx].cpu() if data.target_pixel_values is not None else None
        self.source_image_grid_thw = (
            data.source_image_grid_thw[idx].cpu() if data.source_image_grid_thw is not None else None
        )
        self.target_image_grid_thw = (
            data.target_image_grid_thw[idx].cpu() if data.target_image_grid_thw is not None else None
        )
        self.source_image_position_ids = (
            data.source_image_position_ids[idx].cpu() if data.source_image_position_ids is not None else None
        )
        self.target_image_position_ids = (
            data.target_image_position_ids[idx].cpu() if data.target_image_position_ids is not None else None
        )
        self.source_mm_token_type_ids = (
            data.source_mm_token_type_ids[idx].cpu() if data.source_mm_token_type_ids is not None else None
        )
        self.target_mm_token_type_ids = (
            data.target_mm_token_type_ids[idx].cpu() if data.target_mm_token_type_ids is not None else None
        )

        self.source_last_token_index = self.source_input_ids.shape[1] - 1
        self.target_last_token_index = self.target_input_ids.shape[1] - 1

        # Store the original number of samples
        self.original_len = self.target_input_ids.shape[0]

    def __len__(self) -> int:
        # Double the length to accommodate swapped pairs
        return 2 * self.original_len

    def __getitem__(self, i: int):
        base_idx = i // 2
        # Even index -> swapped, odd index -> original
        swap = i % 2 == 0

        if swap:
            # Original order: source remains source, target remains target
            item = {
                "input_ids": self.target_input_ids[base_idx],
                "attention_mask": self.target_attention_mask[base_idx],
                "source_input_ids": self.source_input_ids[base_idx],
                "source_attention_mask": self.source_attention_mask[base_idx],
                "labels": torch.stack(
                    [
                        self.source_answer_index[base_idx],
                        self.target_answer_index[base_idx],
                    ],
                    dim=0,
                ),
                "sources->base": torch.tensor(
                    [self.target_last_token_index],
                    dtype=torch.long,
                ),
                "subspaces": torch.tensor(0, dtype=torch.long),
            }
            if self.target_pixel_values is not None:
                item["pixel_values"] = self.target_pixel_values[base_idx]
                item["source_pixel_values"] = self.source_pixel_values[base_idx]
            if self.target_image_grid_thw is not None:
                item["image_grid_thw"] = self.target_image_grid_thw[base_idx]
                item["source_image_grid_thw"] = self.source_image_grid_thw[base_idx]
            if self.target_image_position_ids is not None:
                item["image_position_ids"] = self.target_image_position_ids[base_idx]
                item["source_image_position_ids"] = self.source_image_position_ids[base_idx]
            if self.target_mm_token_type_ids is not None:
                item["mm_token_type_ids"] = self.target_mm_token_type_ids[base_idx]
                item["source_mm_token_type_ids"] = self.source_mm_token_type_ids[base_idx]
            return item
        else:
            # Swapped order: target becomes source, source becomes target
            item = {
                "input_ids": self.source_input_ids[base_idx],
                "attention_mask": self.source_attention_mask[base_idx],
                "source_input_ids": self.target_input_ids[base_idx],
                "source_attention_mask": self.target_attention_mask[base_idx],
                "labels": torch.stack(
                    [
                        self.target_answer_index[base_idx],
                        self.source_answer_index[base_idx],
                    ],
                    dim=0,
                ),
                "sources->base": torch.tensor(
                    [self.source_last_token_index],
                    dtype=torch.long,
                ),
                "subspaces": torch.tensor(0, dtype=torch.long),
            }
            if self.source_pixel_values is not None:
                item["pixel_values"] = self.source_pixel_values[base_idx]
                item["source_pixel_values"] = self.target_pixel_values[base_idx]
            if self.source_image_grid_thw is not None:
                item["image_grid_thw"] = self.source_image_grid_thw[base_idx]
                item["source_image_grid_thw"] = self.target_image_grid_thw[base_idx]
            if self.source_image_position_ids is not None:
                item["image_position_ids"] = self.source_image_position_ids[base_idx]
                item["source_image_position_ids"] = self.target_image_position_ids[base_idx]
            if self.source_mm_token_type_ids is not None:
                item["mm_token_type_ids"] = self.source_mm_token_type_ids[base_idx]
                item["source_mm_token_type_ids"] = self.target_mm_token_type_ids[base_idx]
            return item


class DASFullAnswerDataset(Dataset):
    """Full-answer probability DAS rows with an intervention at every answer step.

    Each row contains a target-intent prompt as the base, the opposite-intent
    prompt as the activation source, and a shared candidate-answer prefix.
    Expanding answers token by token matches autoregressive steering, where the
    last-token hook is applied again at every generation step.
    """

    def __init__(
        self,
        data: PatchingData,
        source_answer_ids: Sequence[Sequence[int]],
        target_answer_ids: Sequence[Sequence[int]],
        *,
        include_alternatives: bool,
    ) -> None:
        n_pairs = data.source_input_ids.shape[0]
        if len(source_answer_ids) != n_pairs or len(target_answer_ids) != n_pairs:
            raise ValueError("Full-answer token lists must align with PatchingData rows")
        if any(not ids for ids in source_answer_ids) or any(not ids for ids in target_answer_ids):
            raise ValueError("Full-answer token lists must not be empty")

        self.data = PatchingData(
            **{
                name: None if getattr(data, name) is None else getattr(data, name).cpu()
                for name in data.__dataclass_fields__
            }
        )
        self.source_answer_ids = [list(ids) for ids in source_answer_ids]
        self.target_answer_ids = [list(ids) for ids in target_answer_ids]
        self.examples: list[tuple[int, bool, bool, int]] = []

        for pair_index in range(n_pairs):
            # source_is_prior=false: context prompt patches the prior prompt.
            # source_is_prior=True: prior prompt patches the context prompt.
            for source_is_prior in (False, True):
                source_ids = (
                    self.source_answer_ids[pair_index] if source_is_prior else self.target_answer_ids[pair_index]
                )
                target_ids = (
                    self.target_answer_ids[pair_index] if source_is_prior else self.source_answer_ids[pair_index]
                )
                candidates = [(True, source_ids)]
                if include_alternatives:
                    candidates.append((False, target_ids))
                for candidate_is_source, candidate_ids in candidates:
                    for token_index in range(len(candidate_ids)):
                        self.examples.append((pair_index, source_is_prior, candidate_is_source, token_index))

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _prompt_sequence(ids: Tensor, mask: Tensor, prefix: Sequence[int]) -> Tensor:
        prompt = ids[mask.bool()]
        if not prefix:
            return prompt
        return torch.cat((prompt, ids.new_tensor(prefix)))

    def _copy_non_sequence_fields(
        self,
        item: dict[str, Tensor],
        *,
        pair_index: int,
        base_side: str,
        source_side: str,
    ) -> None:
        for name in ("pixel_values", "image_grid_thw", "image_position_ids"):
            base_value = getattr(self.data, f"{base_side}_{name}")
            source_value = getattr(self.data, f"{source_side}_{name}")
            if base_value is not None:
                item[name] = base_value[pair_index]
                item[f"source_{name}"] = source_value[pair_index]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        pair_index, source_is_prior, candidate_is_source, token_index = self.examples[index]
        source_side = "source" if source_is_prior else "target"
        base_side = "target" if source_is_prior else "source"
        source_answer = self.source_answer_ids[pair_index] if source_is_prior else self.target_answer_ids[pair_index]
        target_answer = self.target_answer_ids[pair_index] if source_is_prior else self.source_answer_ids[pair_index]
        candidate = source_answer if candidate_is_source else target_answer
        prefix = candidate[:token_index]

        base_ids = getattr(self.data, f"{base_side}_input_ids")[pair_index]
        base_mask = getattr(self.data, f"{base_side}_attention_mask")[pair_index]
        source_ids = getattr(self.data, f"{source_side}_input_ids")[pair_index]
        source_mask = getattr(self.data, f"{source_side}_attention_mask")[pair_index]
        input_ids = self._prompt_sequence(base_ids, base_mask, prefix)
        source_input_ids = self._prompt_sequence(source_ids, source_mask, prefix)

        item = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "source_input_ids": source_input_ids,
            "source_attention_mask": torch.ones_like(source_input_ids),
            "labels": torch.tensor([candidate[token_index], -100], dtype=torch.long),
            "subspaces": torch.tensor(0, dtype=torch.long),
            "_candidate_is_source": torch.tensor(candidate_is_source, dtype=torch.bool),
            "_preference_group": torch.tensor(pair_index * 2 + int(source_is_prior), dtype=torch.long),
            "_pair_id": torch.tensor(pair_index, dtype=torch.long),
            "_source_is_prior": torch.tensor(source_is_prior, dtype=torch.bool),
        }

        for name in ("mm_token_type_ids",):
            base_value = getattr(self.data, f"{base_side}_{name}")
            source_value = getattr(self.data, f"{source_side}_{name}")
            if base_value is not None:
                base_prompt = base_value[pair_index][base_mask.bool()]
                source_prompt = source_value[pair_index][source_mask.bool()]
                suffix = base_prompt.new_zeros(len(prefix))
                item[name] = torch.cat((base_prompt, suffix))
                item[f"source_{name}"] = torch.cat((source_prompt, suffix.clone()))

        self._copy_non_sequence_fields(
            item,
            pair_index=pair_index,
            base_side=base_side,
            source_side=source_side,
        )
        return item


def full_answer_das_collator(
    batch: Sequence[Mapping[str, Tensor]],
    *,
    pad_token_id: int,
) -> dict[str, Tensor]:
    """Left-pad base/source prefixes to one shared last-token location."""
    sequence_keys = {
        "input_ids": pad_token_id,
        "attention_mask": 0,
        "mm_token_type_ids": 0,
        "source_input_ids": pad_token_id,
        "source_attention_mask": 0,
        "source_mm_token_type_ids": 0,
    }
    max_length = max(max(sample["input_ids"].numel(), sample["source_input_ids"].numel()) for sample in batch)
    collated: dict[str, Tensor] = {}
    handled = set()
    for key, padding_value in sequence_keys.items():
        if key not in batch[0]:
            continue
        rows = []
        for sample in batch:
            row = sample[key]
            padding = row.new_full((max_length - row.numel(),), padding_value)
            rows.append(torch.cat((padding, row)))
        collated[key] = torch.stack(rows)
        handled.add(key)

    for key in batch[0]:
        if key in handled:
            continue
        collated[key] = torch.stack([sample[key] for sample in batch])
    collated["sources->base"] = torch.full((len(batch), 1), max_length - 1, dtype=torch.long)
    return collated


def build_full_answer_das_dataset(
    data: PatchingData,
    source_answer_ids: Sequence[Sequence[int]],
    target_answer_ids: Sequence[Sequence[int]],
    *,
    include_alternatives: bool = False,
) -> DASFullAnswerDataset:
    return DASFullAnswerDataset(
        data,
        source_answer_ids,
        target_answer_ids,
        include_alternatives=include_alternatives,
    )


def build_das_datasets(data: PatchingData):

    if data.source_pixel_values is not None and len(data.source_pixel_values.size()) == 2:
        if data.source_image_grid_thw is not None:
            num_patches = int(torch.prod(data.source_image_grid_thw[0]).item())
            data.source_pixel_values = data.source_pixel_values.view(-1, num_patches, data.source_pixel_values.size(-1))
            data.target_pixel_values = data.target_pixel_values.view(-1, num_patches, data.target_pixel_values.size(-1))
    disagree_mask = data.source_answer_index != data.target_answer_index
    kept_indices = torch.where(disagree_mask)[0]
    if kept_indices.numel() == 0:
        raise ValueError("No usable DAS pairs remain (source/target first token are identical).")

    full_dataset = DASPairDataset(data, kept_indices)

    return full_dataset
