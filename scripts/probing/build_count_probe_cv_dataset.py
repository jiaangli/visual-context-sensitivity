from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_INPUTS = (
    "dataset/whatifvis/count_val.jsonl",
    "dataset/whatifvis/count_test.jsonl",
)
DEFAULT_LABELS = (2, 3, 4, 5)
DEFAULT_GENERIC_QUESTION = (
    'How many target objects are visible in the image? Answer in the format "Count: N", '
    'where N is the count written in decimal digits (e.g., "Count: 9").'
)
DEFAULT_FACTUAL_DIR = "dataset/whatifvis/images/vlms_are_biased_factual"
PAIRED_SOURCE_DIR = "vlms_are_biased"
SHARED_FACTUAL_CONCEPTS = ("adidas", "nike")
CAR_CONCEPTS = ("audi", "maserati", "mercedes_benz")
# The source dataset has no factual photo of this car. Its one other Mercedes-Benz photo shows a different
# car, so it is a factual image of the same concept, not an image-level counterpart.
FACTUAL_STEM_ALIASES = {"mercedes_benz_black_sedan_0": "mercedes"}
# Its one factual image shows a different scene from every counterfactual image, so it has no real pairs.
CONCEPTS_WITHOUT_IMAGE_PAIRS = ("adidas",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_rows(paths: Sequence[Path]) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    source_metadata: dict[str, dict] = {}
    for path in paths:
        resolved_source = path.resolve()
        source_rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                missing = {"question", "image", "context_answer", "prior_answer"} - row.keys()
                if missing:
                    raise ValueError(f"{path}:{line_number} is missing fields: {sorted(missing)}")
                copied = dict(row)
                copied["_source_path"] = str(resolved_source)
                copied["_source_name"] = path.name
                copied["_source_split"] = path.stem.removeprefix("count_")
                copied["_source_line"] = line_number
                rows.append(copied)
                source_rows += 1
        source_metadata[str(path)] = {
            "sha256": sha256_file(path),
            "rows": source_rows,
        }
    return rows, source_metadata


def resolve_image_path(row: dict, dataset_root: Path) -> Path:
    raw_path = Path(str(row["image"]))
    source_parent = Path(row["_source_path"]).parent
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [
            source_parent / raw_path,
            dataset_root / raw_path,
            dataset_root / "whatifvis" / raw_path,
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Image {row['image']!r} was not found; tried {rendered}")


def dataset_relative_image(path: Path, dataset_root: Path) -> str:
    resolved_root = dataset_root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Image {resolved_path} is not under dataset root {resolved_root}") from exc
    return "./" + relative.as_posix()


def concept_and_factual_stem(image_stem: str) -> tuple[str, str]:
    """Map a VLMs-are-Biased counterfactual image stem to its concept and factual image stem."""

    base = image_stem.removesuffix("_768")
    head = base.split("-", 1)[0]
    if head in SHARED_FACTUAL_CONCEPTS or head.startswith("flag_of_"):
        return head, head
    if head in CAR_CONCEPTS:
        return base, f"{base}_0"
    animal = re.sub(r"_\d+_\d+$", "", base)
    return animal, animal


def concept_of(row: dict) -> str:
    """Name the concept a source row shows: an animal species, a car photo, a shoe brand, a flag, or a question."""

    image = Path(str(row["image"]))
    if image.parent.name == PAIRED_SOURCE_DIR:
        return concept_and_factual_stem(image.stem)[0]
    question = re.sub(r"\s*Answer\b.*$", "", str(row["question"])).strip().lower()
    return f"{image.parent.name}:{question}"


def _normalize_stem(stem: str) -> str:
    return stem.lower().replace("-", "_").replace("gray", "grey")


def _representative_index(rows: Sequence[dict]) -> int:
    """Choose one question variant without favoring input order."""

    return random.randrange(len(rows))


def build_unique_image_rows(
    rows: Sequence[dict],
    *,
    dataset_root: Path,
    labels: Sequence[int],
    question_template: str | None = None,
) -> tuple[list[dict], dict]:
    requested_labels = set(labels)
    if not requested_labels:
        raise ValueError("At least one context label is required")

    by_digest: dict[str, list[dict]] = defaultdict(list)
    eligible_source_rows = 0
    for source_row in rows:
        try:
            label = int(str(source_row["context_answer"]).strip())
        except ValueError as exc:
            raise ValueError(
                f"Non-integer context answer at {source_row['_source_name']}:"
                f"{source_row['_source_line']}: {source_row['context_answer']!r}"
            ) from exc
        if label not in requested_labels:
            continue
        eligible_source_rows += 1
        resolved_image = resolve_image_path(source_row, dataset_root)
        enriched = dict(source_row)
        enriched["_resolved_image"] = str(resolved_image)
        enriched["_context_label"] = label
        by_digest[sha256_file(resolved_image)].append(enriched)

    unique_rows: list[dict] = []
    for image_digest, image_rows in sorted(by_digest.items()):
        image_labels = {row["_context_label"] for row in image_rows}
        if len(image_labels) != 1:
            sources = [f"{row['_source_name']}:{row['_source_line']}" for row in image_rows]
            raise ValueError(
                f"Byte-identical image {image_digest} has conflicting labels {sorted(image_labels)}: {sources}"
            )

        ordered = sorted(
            image_rows,
            key=lambda row: (
                row["question"],
                row["_source_name"],
                row["_source_line"],
            ),
        )
        representative = ordered[_representative_index(ordered)]
        output_row = {key: value for key, value in representative.items() if not key.startswith("_")}
        if question_template is not None:
            output_row["question"] = question_template
        label = next(iter(image_labels))
        output_row.update(
            {
                "image": dataset_relative_image(Path(representative["_resolved_image"]), dataset_root),
                "context_answer": str(label),
                "probe_target": float(label),
                "probe_label": label,
                "probe_group_id": f"concept:{concept_of(representative)}",
                "probe_concept": concept_of(representative),
                "probe_image_sha256": image_digest,
                "probe_image_type": "counterfactual",
                "probe_source_split": representative["_source_split"],
                "probe_source_splits": sorted({row["_source_split"] for row in image_rows}),
                "probe_source_rows": [f"{row['_source_name']}:{row['_source_line']}" for row in ordered],
                "probe_collapsed_row_count": len(image_rows),
                "probe_question_variant_count": len({row["question"] for row in image_rows}),
                "probe_sample_weight": 1.0,
            }
        )
        unique_rows.append(output_row)

    label_counts = Counter(row["probe_label"] for row in unique_rows)
    missing_labels = requested_labels - label_counts.keys()
    if missing_labels:
        raise ValueError(f"No image remains for requested labels {sorted(missing_labels)}")

    summary = {
        "eligible_source_rows": eligible_source_rows,
        "unique_image_rows": len(unique_rows),
        "collapsed_duplicate_rows": eligible_source_rows - len(unique_rows),
        "unique_rows_per_label": dict(sorted(label_counts.items())),
        "source_row_multiplicity": dict(
            sorted(Counter(row["probe_collapsed_row_count"] for row in unique_rows).items())
        ),
        "content_duplicate_path_groups": sum(
            len({row["_resolved_image"] for row in image_rows}) > 1 for image_rows in by_digest.values()
        ),
    }
    return unique_rows, summary


def build_factual_rows(
    counterfactual_rows: Sequence[dict],
    *,
    factual_dir: Path,
    dataset_root: Path,
    labels: Sequence[int],
) -> tuple[list[dict], list[dict], dict]:
    """Pair retained counterfactual rows with their factual images.

    A factual image's label is the prior answer of its counterfactual rows, since
    the prior answer is the real-world count that the image shows. Counterfactual
    rows without a factual image inside the requested labels are excluded.

    Returns the paired counterfactual rows, the factual rows, and a summary.
    """

    requested_labels = set(labels)
    factual_paths: dict[str, Path] = {}
    for path in sorted(factual_dir.iterdir()):
        if not path.is_file():
            continue
        key = _normalize_stem(path.stem)
        if key in factual_paths:
            raise ValueError(f"Factual images {factual_paths[key]} and {path} share the key {key!r}")
        factual_paths[key] = path
    if not factual_paths:
        raise FileNotFoundError(f"No factual images found in {factual_dir}")

    rows_by_factual: dict[str, list[dict]] = defaultdict(list)
    without_factual_source = []
    without_image_pair = []
    for row in counterfactual_rows:
        image = Path(str(row["image"]))
        if row["probe_concept"] in CONCEPTS_WITHOUT_IMAGE_PAIRS:
            without_image_pair.append(row["image"])
            continue
        if image.parent.name != PAIRED_SOURCE_DIR:
            without_factual_source.append(row["image"])
            continue
        key = _normalize_stem(concept_and_factual_stem(image.stem)[1])
        key = FACTUAL_STEM_ALIASES.get(key, key)
        if key not in factual_paths:
            raise FileNotFoundError(f"No factual image in {factual_dir} for {row['image']}")
        rows_by_factual[key].append(row)

    paired_rows = []
    factual_rows = []
    label_outside_labels = {}
    for key, paired in sorted(rows_by_factual.items()):
        priors = {str(row["prior_answer"]).strip() for row in paired}
        if len(priors) != 1:
            raise ValueError(f"Factual image {key!r} has conflicting prior answers: {sorted(priors)}")
        label = int(priors.pop())
        if label not in requested_labels:
            label_outside_labels[key] = {"label": label, "counterfactual_images": [row["image"] for row in paired]}
            continue

        paired_rows.extend(paired)
        ordered = sorted(paired, key=lambda row: (row["question"], row["image"]))
        image_digest = sha256_file(factual_paths[key])
        factual_rows.append(
            {
                "question": ordered[0]["question"],
                "image": dataset_relative_image(factual_paths[key], dataset_root),
                "context_answer": str(label),
                "prior_answer": str(label),
                "response_template": ordered[0].get("response_template"),
                "statement": None,
                "probe_target": float(label),
                "probe_label": label,
                "probe_group_id": f"concept:{ordered[0]['probe_concept']}",
                "probe_concept": ordered[0]["probe_concept"],
                "probe_image_sha256": image_digest,
                "probe_image_type": "factual",
                "probe_source_split": "factual",
                "probe_source_splits": ["factual"],
                "probe_source_rows": sorted(source for row in paired for source in row["probe_source_rows"]),
                "probe_collapsed_row_count": 1,
                "probe_question_variant_count": 1,
                "probe_sample_weight": 1.0,
            }
        )

    if not factual_rows:
        raise ValueError(f"No factual image from {factual_dir} has a requested label")
    summary = {
        "factual_dir": str(factual_dir),
        "label_rule": "the prior answer of the paired counterfactual rows",
        "concept_level_pairs": FACTUAL_STEM_ALIASES,
        "factual_images_available": len(factual_paths),
        "factual_rows": len(factual_rows),
        "factual_rows_per_label": dict(sorted(Counter(row["probe_label"] for row in factual_rows).items())),
        "factual_images_without_retained_counterfactual": sorted(factual_paths.keys() - rows_by_factual.keys()),
        "counterfactual_rows": len(paired_rows),
        "excluded_counterfactual_rows": {
            "no_factual_source": sorted(without_factual_source),
            "no_image_level_pair": sorted(without_image_pair),
            "factual_label_outside_requested_labels": label_outside_labels,
        },
    }
    return paired_rows, factual_rows, summary


def assign_stratified_group_folds(
    rows: Sequence[dict],
    *,
    n_splits: int,
) -> list[dict]:
    """Assign whole concepts to folds, keeping each label as evenly spread as the concepts allow."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    rows_by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_group[row["probe_group_id"]].append(dict(row))

    groups_per_label: dict[int, set[str]] = defaultdict(set)
    for group, group_rows in rows_by_group.items():
        for row in group_rows:
            groups_per_label[row["probe_label"]].add(group)
    too_small = {label: len(groups) for label, groups in groups_per_label.items() if len(groups) < n_splits}
    if too_small:
        raise ValueError(f"Every retained label needs at least {n_splits} concepts; found {too_small}")

    label_totals = Counter(row["probe_label"] for row in rows)
    fold_label_counts = [Counter() for _ in range(n_splits)]
    fold_totals = [0] * n_splits

    def label_spread(fold: int, group_counts: Counter) -> float:
        """Mean, over labels, of the variance of each fold's share of that label after adding a group."""
        spread = 0.0
        for label, total in label_totals.items():
            shares = [
                (fold_label_counts[index][label] + (group_counts[label] if index == fold else 0)) / total
                for index in range(n_splits)
            ]
            mean_share = sum(shares) / n_splits
            spread += sum((share - mean_share) ** 2 for share in shares) / n_splits
        return spread / len(label_totals)

    # Place large concepts first, because they constrain the balance most.
    group_order = sorted(rows_by_group)
    random.shuffle(group_order)
    group_order.sort(key=lambda group: len(rows_by_group[group]), reverse=True)

    assigned = []
    for group in group_order:
        group_rows = rows_by_group[group]
        group_counts = Counter(row["probe_label"] for row in group_rows)
        tie_order = list(range(n_splits))
        random.shuffle(tie_order)
        fold = min(
            range(n_splits),
            key=lambda index: (round(label_spread(index, group_counts), 12), fold_totals[index], tie_order[index]),
        )
        for row in group_rows:
            row["probe_cv_fold"] = fold
            row["probe_split"] = "cv"
            assigned.append(row)
        fold_label_counts[fold].update(group_counts)
        fold_totals[fold] += len(group_rows)

    return sorted(assigned, key=lambda row: (row["probe_cv_fold"], row["probe_image_sha256"]))


def _with_outer_split(rows: Sequence[dict], split: str, outer_fold: int) -> list[dict]:
    copied = []
    for row in rows:
        output_row = dict(row)
        output_row["probe_split"] = split
        output_row["probe_outer_fold"] = outer_fold
        copied.append(output_row)

    for row in copied:
        row["probe_sample_weight"] = 1.0
    return copied


def validate_cv_rows(rows: Sequence[dict], n_splits: int) -> None:
    hashes = [row["probe_image_sha256"] for row in rows]
    image_paths = [row["image"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise AssertionError("Cross-validation rows are not unique by image content")
    if len(image_paths) != len(set(image_paths)):
        raise AssertionError("Cross-validation rows are not unique by image path")
    if {row["probe_cv_fold"] for row in rows} != set(range(n_splits)):
        raise AssertionError("Not every requested fold received at least one image")

    by_label: dict[int, list[int]] = defaultdict(lambda: [0] * n_splits)
    for row in rows:
        by_label[row["probe_label"]][row["probe_cv_fold"]] += 1

    folds_by_group: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        folds_by_group[row["probe_group_id"]].add(row["probe_cv_fold"])
    split_groups = {group: sorted(folds) for group, folds in folds_by_group.items() if len(folds) > 1}
    if split_groups:
        raise AssertionError(f"Concepts span more than one fold: {split_groups}")
    for label, counts in by_label.items():
        if 0 in counts:
            raise AssertionError(f"Label {label} is missing from at least one fold: {counts}")

    types_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        types_by_group[row["probe_group_id"]].add(row["probe_image_type"])
    unpaired_groups = sorted(group for group, types in types_by_group.items() if len(types) != 2)
    if unpaired_groups:
        raise AssertionError(f"Concepts lack a factual or counterfactual image: {unpaired_groups}")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_and_write(
    *,
    input_paths: Sequence[Path],
    dataset_root: Path,
    output_dir: Path,
    labels: Sequence[int],
    n_splits: int,
    seed: int,
    factual_dir: Path,
    question_template: str | None = None,
) -> dict:
    resolved_inputs = {path.resolve() for path in input_paths}
    if any(output_dir.resolve() == path or output_dir.resolve() in path.parents for path in resolved_inputs):
        raise ValueError("Output directory must not contain or replace an input source file")

    rows, source_metadata = load_source_rows(input_paths)
    original_hashes = {path: metadata["sha256"] for path, metadata in source_metadata.items()}
    unique_rows, deduplication = build_unique_image_rows(
        rows,
        dataset_root=dataset_root,
        labels=labels,
        question_template=question_template,
    )
    counterfactual_rows, factual_rows, pairing = build_factual_rows(
        unique_rows,
        factual_dir=factual_dir,
        dataset_root=dataset_root,
        labels=labels,
    )
    if question_template is not None:
        factual_rows = [{**row, "question": question_template} for row in factual_rows]
    assigned_rows = assign_stratified_group_folds(counterfactual_rows + factual_rows, n_splits=n_splits)
    validate_cv_rows(assigned_rows, n_splits)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "count_cv_unique_all.jsonl"
    write_jsonl(all_path, assigned_rows)

    fold_summaries = []
    for outer_fold in range(n_splits):
        test_base = [row for row in assigned_rows if row["probe_cv_fold"] == outer_fold]
        train_base = [row for row in assigned_rows if row["probe_cv_fold"] != outer_fold]
        train_rows = _with_outer_split(train_base, "train", outer_fold)
        test_rows = _with_outer_split(test_base, "test", outer_fold)

        train_hashes = {row["probe_image_sha256"] for row in train_rows}
        test_hashes = {row["probe_image_sha256"] for row in test_rows}
        if train_hashes & test_hashes:
            raise AssertionError(f"Image-content leakage in outer fold {outer_fold}")
        train_groups = {row["probe_group_id"] for row in train_rows}
        test_groups = {row["probe_group_id"] for row in test_rows}
        if train_groups & test_groups:
            raise AssertionError(f"Concept leakage in outer fold {outer_fold}: {sorted(train_groups & test_groups)}")

        fold_dir = output_dir / f"fold_{outer_fold}"
        train_path = fold_dir / "count_cv_train.jsonl"
        test_path = fold_dir / "count_cv_test.jsonl"
        write_jsonl(train_path, train_rows)
        write_jsonl(test_path, test_rows)
        fold_summaries.append(
            {
                "outer_fold": outer_fold,
                "train_file": str(train_path),
                "test_file": str(test_path),
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "train_rows_per_label": dict(sorted(Counter(row["probe_label"] for row in train_rows).items())),
                "test_rows_per_label": dict(sorted(Counter(row["probe_label"] for row in test_rows).items())),
                "test_rows_per_image_type": dict(
                    sorted(Counter(row["probe_image_type"] for row in test_rows).items())
                ),
                "test_concepts": len(test_groups),
                "test_rows_per_source_split": dict(
                    sorted(Counter(row["probe_source_split"] for row in test_rows).items())
                ),
            }
        )

    current_hashes = {str(path): sha256_file(path) for path in input_paths}
    if current_hashes != original_hashes:
        raise RuntimeError("A read-only source JSONL changed while building the CV dataset")

    manifest = {
        "schema_version": 1,
        "construction": "concept-grouped stratified cross-validation over counterfactual and factual images",
        "seed": seed,
        "n_splits": n_splits,
        "labels": sorted(set(labels)),
        "source_files": source_metadata,
        "source_rows": len(rows),
        "deduplication": {
            "key": "SHA-256 of resolved image bytes",
            "representative_question": (
                "one fixed generic question shared by every unique image"
                if question_template is not None
                else "one deterministic seed-selected question per unique image"
            ),
            **deduplication,
        },
        "pairing": pairing,
        "rows_per_image_type": dict(sorted(Counter(row["probe_image_type"] for row in assigned_rows).items())),
        "rows_per_label": dict(sorted(Counter(row["probe_label"] for row in assigned_rows).items())),
        "question_policy": {
            "mode": "fixed_generic" if question_template is not None else "source_representative",
            "template": question_template,
            "unique_emitted_questions": len({row["question"] for row in assigned_rows}),
        },
        "all_file": str(all_path),
        "folds": fold_summaries,
        "evaluation_protocol": {
            "outer": (
                f"{n_splits}-fold stratified cross-validation; each unique image is held out exactly once; "
                "all images of a concept share one fold"
            ),
            "fit": (
                "fit the same preregistered linear-probe configuration on the other folds; "
                "do not select hyperparameters or layers from held-out predictions"
            ),
            "aggregation": "pool out-of-fold predictions and report macro metrics with image-level bootstrap",
        },
        "notes": [
            "The source count_val.jsonl and count_test.jsonl files are read-only and hash-checked.",
            "Every emitted JSONL contains at most one row per image path and per image-content hash.",
            "The same image can appear in training files for other outer folds, as required by cross-validation.",
            "Every concept has at least one counterfactual and one factual image.",
            "A concept never appears in both the training and test files of an outer fold.",
            "Whole concepts are assigned to folds, so per-label fold counts are only approximately equal.",
            "Class balancing belongs to the probe estimator and is not encoded as fold-specific dataset weights.",
            "The no-image and shuffled-image controls must reuse these exact fold assignments.",
            (
                "Every row uses the same generic question, so question wording and textual object "
                "identity cannot predict the probe label."
                if question_template is not None
                else "Question wording is retained from a deterministic source representative."
            ),
        ],
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=list(DEFAULT_INPUTS))
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-dir", default="dataset/probing/count_cv_unique")
    parser.add_argument("--labels", nargs="+", type=int, default=list(DEFAULT_LABELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generic-question",
        action="store_true",
        help="Replace every emitted question with one fixed, object-agnostic template",
    )
    parser.add_argument(
        "--factual-dir",
        default=DEFAULT_FACTUAL_DIR,
        help="Directory of factual images that are paired with VLMs-are-Biased counterfactual images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    manifest = build_and_write(
        input_paths=[Path(path) for path in args.inputs],
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        labels=args.labels,
        n_splits=args.folds,
        seed=args.seed,
        factual_dir=Path(args.factual_dir),
        question_template=DEFAULT_GENERIC_QUESTION if args.generic_question else None,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
