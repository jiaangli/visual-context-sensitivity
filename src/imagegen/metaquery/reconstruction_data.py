from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color

TEST_SET_NAMES = ("color", "count", "size", "spatial_temporal", "weight")


@dataclass(frozen=True)
class ReconstructionRecord:
    """One unique source image and its task metadata."""

    task: str
    row_index: int
    input_path: str
    input_path_raw: str
    output_name: str
    question: str
    context_answer: str
    prior_answer: str
    statement: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_image_path(dataset_root: str | Path, image_path: str) -> Path:
    """Resolve a WhatIfVis image path (lower-cased, spaces replaced by underscores)."""
    joined = Path(dataset_root) / image_path.lstrip("./")
    return Path(str(joined).lower().replace(" ", "_"))


def load_unique_records(
    dataset_root: str | Path,
    dataset_file: str | Path,
    task: str,
) -> list[ReconstructionRecord]:
    """Load all unique images from a test JSONL without source filtering."""
    if task not in TEST_SET_NAMES:
        raise ValueError(f"unknown task {task!r}; expected one of {TEST_SET_NAMES}")

    records: list[ReconstructionRecord] = []
    seen_paths: set[str] = set()
    seen_output_names: dict[str, str] = {}
    dataset_path = Path(dataset_file)

    with dataset_path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            input_path = normalize_image_path(dataset_root, row["image"])
            normalized = str(input_path)
            if normalized in seen_paths:
                continue
            if not input_path.is_file():
                raise FileNotFoundError(f"missing input image referenced by {dataset_path}: {input_path}")

            output_name = f"{input_path.stem}_generated{input_path.suffix.lower()}"
            previous = seen_output_names.get(output_name)
            if previous is not None and previous != normalized:
                raise ValueError(
                    f"output filename collision in {dataset_path}: {previous!r} and "
                    f"{normalized!r} both map to {output_name!r}"
                )

            seen_paths.add(normalized)
            seen_output_names[output_name] = normalized
            records.append(
                ReconstructionRecord(
                    task=task,
                    row_index=row_index,
                    input_path=normalized,
                    input_path_raw=str(row["image"]),
                    output_name=output_name,
                    question=str(row.get("question", "")),
                    context_answer=str(row.get("context_answer", "")),
                    prior_answer=str(row.get("prior_answer", "")),
                    statement=str(row.get("statement", "")),
                )
            )

    return records


def load_test_suite(
    dataset_root: str | Path,
    tasks: Sequence[str] = TEST_SET_NAMES,
) -> dict[str, list[ReconstructionRecord]]:
    """Load requested ``<task>_test.jsonl`` files from ``dataset_root``."""
    root = Path(dataset_root)
    return {task: load_unique_records(root, root / f"{task}_test.jsonl", task) for task in tasks}


def stable_image_seed(base_seed: int, task: str, input_path: str) -> int:
    """Return a stable per-image seed so retries do not change later samples."""
    digest = hashlib.blake2b(f"{task}\0{input_path}".encode(), digest_size=8).digest()
    offset = int.from_bytes(digest, "big")
    return (int(base_seed) + offset) % (2**63 - 1)


def color_correct(generated_img: Image.Image, reference_img: Image.Image) -> Image.Image:
    """Match Lab channel means/stds, reproducing the paper's post-processing."""
    generated_rgb = np.asarray(generated_img.convert("RGB"), dtype=np.float32) / 255.0
    reference_rgb = np.asarray(reference_img.convert("RGB"), dtype=np.float32) / 255.0
    generated_lab = color.rgb2lab(generated_rgb)
    reference_lab = color.rgb2lab(reference_rgb)
    corrected_lab = np.zeros_like(generated_lab)
    for channel in range(3):
        generated_mean = generated_lab[..., channel].mean()
        generated_std = generated_lab[..., channel].std()
        reference_mean = reference_lab[..., channel].mean()
        reference_std = reference_lab[..., channel].std()
        if generated_std > 1e-5:
            corrected_lab[..., channel] = (generated_lab[..., channel] - generated_mean) * (
                reference_std / generated_std
            ) + reference_mean
        else:
            corrected_lab[..., channel] = generated_lab[..., channel] - generated_mean + reference_mean
    corrected_rgb = np.clip(color.lab2rgb(corrected_lab), 0.0, 1.0)
    return Image.fromarray((corrected_rgb * 255.0).astype(np.uint8))


def is_count_animal(record: ReconstructionRecord) -> bool:
    """Whether a count example is one of the paired animal counterfactuals."""
    path = Path(record.input_path)
    return (
        record.task == "count" and path.parent.name == "vlms_are_biased" and not path.stem.lower().startswith("flag_")
    )


def count_factual_path(
    record: ReconstructionRecord,
    factual_dir: str | Path,
) -> Path:
    """Resolve the natural-image counterpart of a count animal example."""
    if not is_count_animal(record):
        raise ValueError(f"not a paired count-animal record: {record.input_path}")
    base_stem = re.sub(r"_\d+_\d+_768$", "", Path(record.input_path).stem)
    matches = [
        candidate for candidate in Path(factual_dir).iterdir() if candidate.is_file() and candidate.stem == base_stem
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one factual image for {record.input_path}, found {len(matches)}")
    return matches[0]


def _clean_prompt(value: str) -> str:
    value = value.strip().strip("'?.!,'\"")
    value = re.sub(r"^(?:a|an|the)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return value


def _question_subject(question: str) -> str:
    patterns = (
        r"(?:main )?color (?:of|is|are)\s+(.+?)(?:\?|$)",
        r"(?:legs|arms|branches) (?:does|of)\s+(?:a|an|the)?\s*(.+?)\s+(?:have|have\?|$)",
        r"(?:legs|arms|branches) of\s+(?:a|an|the)?\s*(.+?)(?:\.|\?|$)",
        r"which is (?:larger|heavier),\s+(.+?)(?:\?|$)",
        r"in which city is\s+(.+?)(?:\s+located)?(?:\?|$)",
        r"what does\s+(?:a|an|the)?\s*(.+?)\s+(?:eat|use|drink)(?:\?|$)",
        r"(?:shape|texture) of\s+(?:a|an|the)?\s*(.+?)(?:\?|$)",
        r"what is (?:behind|covering)\s+(?:a|an|the)?\s*(.+?)(?:\?|$)",
    )
    lowered = question.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            subject = re.split(r"\s+or\s+|,", match.group(1), maxsplit=1)[0]
            return _clean_prompt(subject)
    return ""


def sam_prompts(record: ReconstructionRecord) -> tuple[str, ...]:
    """Return the task target used for SAM3 foreground mIoU.

    Size and weight use the counterfactual-answer object. Other tasks use the
    object named by the question. Keeping one prompt per image makes mIoU
    comparable and avoids turning the metric into a union of unrelated masks.
    """
    path = Path(record.input_path)
    if is_count_animal(record):
        return ("animal",)
    if record.task == "count" and path.stem.lower().startswith("flag_"):
        return ("flag",)

    subject = _question_subject(record.question)
    if record.task in {"size", "weight"}:
        target = _clean_prompt(record.context_answer)
        if target:
            return (target,)
    if subject:
        return (subject,)
    target = _clean_prompt(record.context_answer)
    if target and not target.isdigit() and target.lower() not in {"yes", "no"}:
        return (target,)
    return ("foreground object",)


def deterministic_sample(
    records: Sequence[ReconstructionRecord],
    count: int,
) -> list[ReconstructionRecord]:
    """Select a random subset from the global RNG and return it in source order."""
    if count <= 0 or count >= len(records):
        return list(records)
    import random

    selected = set(random.sample(range(len(records)), count))
    return [record for index, record in enumerate(records) if index in selected]


def records_to_jsonl(records: Iterable[dict[str, object]], path: str | Path) -> None:
    """Atomically write JSONL records."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(destination)
