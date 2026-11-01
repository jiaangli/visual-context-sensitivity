"""Representation extraction shared by the paper's Count linear probe."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.sft.data_process import (
    SYSTEM_PROMPT,
    CustomCollator,
    DataEntry,
    Sample,
    SampleMetadata,
    build_sample,
    preprocess,
)

logger = logging.getLogger(__name__)
CONDITIONS = ("image", "no_image", "shuffled_image")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _derange_images(images: Sequence[str]) -> dict[str, str]:
    if len(images) < 2:
        raise ValueError("Need at least two images to construct shuffled-image control")
    sources = sorted(images)
    # Sattolo's algorithm produces one cycle, hence no image maps to itself.
    for index in range(len(sources) - 1, 0, -1):
        swap_index = random.randrange(index)
        sources[index], sources[swap_index] = sources[swap_index], sources[index]
    mapping = dict(zip(sorted(images), sources))
    if any(target == source for target, source in mapping.items()):
        raise AssertionError("Shuffled-image mapping retained an original image")
    return mapping


def build_shuffled_image_map(
    rows: Sequence[dict],
    group_key: str | None = None,
) -> dict[str, str]:
    """Randomly permute unique images while preventing identity mappings.

    The permutation is intentionally label-agnostic. Forcing every replacement
    to have a different count would induce an artificial anti-correlation,
    especially in the test split where labels 4 and 5 dominate.

    When ``group_key`` is provided, images are deranged independently within
    each group. Cross-validation uses this to prevent a held-out row from being
    paired with an image belonging to a training fold.
    """
    image_labels: dict[str, int] = {}
    for row in rows:
        image = row["image"]
        label = int(row["probe_label"])
        previous = image_labels.setdefault(image, label)
        if previous != label:
            raise ValueError(f"Image {image!r} has conflicting labels")

    if group_key is None:
        return _derange_images(sorted(image_labels))

    rows_by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if group_key not in row:
            raise ValueError(f"Row is missing shuffle group key {group_key!r}")
        rows_by_group[str(row[group_key])].append(row)

    mapping: dict[str, str] = {}
    for group in sorted(rows_by_group):
        group_images = sorted({row["image"] for row in rows_by_group[group]})
        overlap = mapping.keys() & set(group_images)
        if overlap:
            raise ValueError(f"Images occur in multiple shuffle groups: {sorted(overlap)[:3]}")
        mapping.update(_derange_images(group_images))
    return mapping


def _question_only_sample(row: dict) -> Sample:
    response_prefix = "Count:"
    return Sample(
        messages=[
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": f"Question: {row['question']}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": response_prefix}]},
        ],
        metadata=SampleMetadata(
            context_answer=str(row["context_answer"]),
            prior_answer=str(row["prior_answer"]),
            mode="probe",
            question=row["question"],
            dataset_name="count_regression",
            response_prefix=response_prefix,
            statement=None,
            image_path=None,
        ),
    )


class CountProbeDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict],
        processor,
        condition: str,
        image_root: str,
        max_length: int,
        shuffle_group_key: str | None = None,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition {condition!r}")
        self.rows = list(rows)
        self.processor = processor
        self.condition = condition
        self.image_root = image_root
        self.max_length = max_length
        self.shuffled_images = (
            build_shuffled_image_map(self.rows, group_key=shuffle_group_key)
            if condition == "shuffled_image"
            else {}
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        if self.condition == "no_image":
            sample = _question_only_sample(row)
        else:
            image = row["image"]
            if self.condition == "shuffled_image":
                image = self.shuffled_images[image]
            entry = DataEntry(
                question=row["question"],
                image=image,
                context_answer=str(row["context_answer"]),
                prior_answer=str(row["prior_answer"]),
                response_template="Count: {}",
                statement=None,
            )
            sample = build_sample(
                entry,
                mode="context",
                dataset_name="count_regression",
                image_root=self.image_root,
                training=False,
                use_image=True,
                use_instruction=False,
            )

        encoded = preprocess(sample, self.processor, training=False, max_length=self.max_length)
        encoded["probe_target"] = torch.tensor([float(row["probe_target"])], dtype=torch.float32)
        encoded["probe_sample_weight"] = torch.tensor([float(row.get("probe_sample_weight", 1.0))], dtype=torch.float32)
        encoded["probe_group_id"] = row["probe_group_id"]
        encoded["probe_image_type"] = row["probe_image_type"]
        encoded["probe_image"] = row["image"]
        encoded["probe_question"] = row["question"]
        return encoded


def _model_inputs(batch: dict, device: torch.device) -> tuple[dict, dict]:
    metadata_keys = {
        "_metadata",
        "probe_target",
        "probe_sample_weight",
        "probe_group_id",
        "probe_image_type",
        "probe_image",
        "probe_question",
    }
    metadata = {key: batch[key] for key in metadata_keys if key in batch}
    inputs = {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if key not in metadata_keys
    }
    return inputs, metadata


def _pad_rows_to_full_batch(rows: Sequence[dict], batch_size: int) -> tuple[list[dict], int]:
    """Pad with disposable row copies so every extraction batch has one shape."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    padded = list(rows)
    if not padded:
        raise ValueError("Cannot extract representations from an empty dataset")
    pad_count = (-len(padded)) % batch_size
    padded.extend(dict(rows[index % len(rows)]) for index in range(pad_count))
    return padded, pad_count


def extract_representations(
    model,
    processor,
    rows: Sequence[dict],
    split: str,
    condition: str,
    output_path: Path,
    image_root: str,
    batch_size: int,
    max_length: int,
    num_workers: int,
    shuffle_group_key: str | None = None,
    pad_to_full_batch: bool = False,
) -> None:
    original_row_count = len(rows)
    extraction_rows, pad_count = _pad_rows_to_full_batch(rows, batch_size) if pad_to_full_batch else (list(rows), 0)
    if pad_count:
        logger.info(
            "Padding representation extraction with %s disposable rows for fixed batch size %s",
            pad_count,
            batch_size,
        )
    dataset = CountProbeDataset(
        extraction_rows,
        processor,
        condition,
        image_root,
        max_length,
        shuffle_group_key=shuffle_group_key,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=CustomCollator(processor, max_length=max_length, inference_mode=True),
        drop_last=pad_to_full_batch,
    )

    representations = []
    targets = []
    sample_weights = []
    group_ids: list[str] = []
    image_types: list[str] = []
    images: list[str] = []
    questions: list[str] = []
    device = model.device

    for batch in tqdm(loader, desc=f"extract {condition}/{split}"):
        inputs, metadata = _model_inputs(batch, device)
        if not bool(inputs["attention_mask"][:, -1].all()):
            raise AssertionError("The final sequence position is padded; [:, -1] is not the answer position")

        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        batch_representations = torch.stack(
            [hidden[:, -1, :].detach().to(torch.float32).cpu() for hidden in hidden_states], dim=1
        )
        representations.append(batch_representations.numpy().astype(np.float16))
        targets.append(metadata["probe_target"].cpu().numpy())
        sample_weights.append(metadata["probe_sample_weight"].cpu().numpy())
        group_ids.extend(metadata["probe_group_id"])
        image_types.extend(metadata["probe_image_type"])
        images.extend(metadata["probe_image"])
        questions.extend(metadata["probe_question"])
        del outputs, hidden_states, batch_representations

    output_path.parent.mkdir(parents=True, exist_ok=True)
    representation_array = np.concatenate(representations, axis=0)[:original_row_count]
    target_array = np.concatenate(targets).astype(np.float32)[:original_row_count]
    weight_array = np.concatenate(sample_weights).astype(np.float32)[:original_row_count]
    np.savez_compressed(
        output_path,
        representations=representation_array,
        targets=target_array,
        sample_weights=weight_array,
        group_ids=np.asarray(group_ids[:original_row_count]),
        image_types=np.asarray(image_types[:original_row_count]),
        images=np.asarray(images[:original_row_count]),
        questions=np.asarray(questions[:original_row_count]),
        condition=np.asarray(condition),
        split=np.asarray(split),
    )
    logger.info("Saved %s representations to %s", original_row_count, output_path)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def validate_representation_cache(path: Path, condition: str) -> None:
    """Reject non-finite or suspiciously duplicated hidden-state trajectories.

    Exact duplicates are expected for repeated no-image prompts, but not when
    the prompt text differs or when two image-conditioned rows use different
    images. This catches silent batching/attention failures before probe fitting.
    """
    cache = load_cache(path)
    representations = cache["representations"]
    if not np.isfinite(representations).all():
        raise RuntimeError(f"Non-finite representation found in {path}")

    questions = cache["questions"].astype(str)
    images = cache["images"].astype(str)
    if condition == "no_image" and len(set(questions.tolist())) == 1:
        reference = np.broadcast_to(representations[:1], representations.shape)
        if not np.array_equal(representations, reference):
            max_delta = float(np.max(np.abs(representations.astype(np.float32) - reference.astype(np.float32))))
            raise RuntimeError(
                "A fixed no-image prompt produced batch-dependent representations "
                f"in {path} (maximum absolute difference {max_delta:g})"
            )

    seen: dict[bytes, int] = {}
    suspicious: list[tuple[int, int]] = []
    for index, representation in enumerate(representations):
        fingerprint = hashlib.sha256(np.ascontiguousarray(representation).tobytes()).digest()
        previous = seen.get(fingerprint)
        if previous is None:
            seen[fingerprint] = index
            continue
        different_prompt = questions[previous] != questions[index]
        different_image = condition != "no_image" and images[previous] != images[index]
        if different_prompt or different_image:
            suspicious.append((previous, index))
            if len(suspicious) == 5:
                break

    if suspicious:
        details = "; ".join(
            f"rows {left}/{right}: {questions[left]!r} vs {questions[right]!r}, "
            f"images {images[left]!r} vs {images[right]!r}"
            for left, right in suspicious
        )
        raise RuntimeError(f"Suspicious exact hidden-state duplicates in {path}: {details}")
