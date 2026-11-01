from __future__ import annotations

import argparse
import hashlib
import json
import random
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
                "probe_group_id": f"sha256:{image_digest}",
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


def assign_stratified_folds(
    rows: Sequence[dict],
    *,
    n_splits: int,
) -> list[dict]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    rows_by_label: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_label[int(row["probe_label"])].append(dict(row))
    too_small = {label: len(items) for label, items in rows_by_label.items() if len(items) < n_splits}
    if too_small:
        raise ValueError(f"Every retained label needs at least {n_splits} unique images; found {too_small}")

    fold_rows: list[list[dict]] = [[] for _ in range(n_splits)]
    fold_totals = [0] * n_splits

    for label in sorted(rows_by_label):
        label_rows = sorted(rows_by_label[label], key=lambda row: row["probe_image_sha256"])
        random.shuffle(label_rows)
        base, remainder = divmod(len(label_rows), n_splits)
        tie_order = list(range(n_splits))
        random.shuffle(tie_order)
        tie_rank = {fold: rank for rank, fold in enumerate(tie_order)}
        allocation_order = sorted(range(n_splits), key=lambda fold: (fold_totals[fold], tie_rank[fold]))
        fold_sizes = {fold: base for fold in range(n_splits)}
        for fold in allocation_order[:remainder]:
            fold_sizes[fold] += 1

        offset = 0
        for fold in allocation_order:
            count = fold_sizes[fold]
            for row in label_rows[offset : offset + count]:
                row["probe_cv_fold"] = fold
                row["probe_split"] = "cv"
                fold_rows[fold].append(row)
            offset += count
            fold_totals[fold] += count

    assigned = [row for rows_in_fold in fold_rows for row in rows_in_fold]
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
    for label, counts in by_label.items():
        if max(counts) - min(counts) > 1:
            raise AssertionError(f"Label {label} is not balanced across folds: {counts}")


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
    assigned_rows = assign_stratified_folds(unique_rows, n_splits=n_splits)
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
        "construction": "image-content-unique stratified cross-validation",
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
        "question_policy": {
            "mode": "fixed_generic" if question_template is not None else "source_representative",
            "template": question_template,
            "unique_emitted_questions": len({row["question"] for row in assigned_rows}),
        },
        "all_file": str(all_path),
        "folds": fold_summaries,
        "evaluation_protocol": {
            "outer": (f"{n_splits}-fold stratified cross-validation; each unique image is held out exactly once"),
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
        question_template=DEFAULT_GENERIC_QUESTION if args.generic_question else None,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
