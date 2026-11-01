from __future__ import annotations

import argparse
import hashlib
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from src.eval_results import (
    add_eval_result_args,
    configure_run_logging,
    prepare_eval_run_from_args,
    write_json_result,
)
from src.model_utils import load_model_and_processor, setup_seed
from src.probing.count_features import (
    CONDITIONS,
    extract_representations,
    load_cache,
    load_jsonl,
    validate_representation_cache,
)

logger = logging.getLogger(__name__)
DEFAULT_LABELS = (2, 3, 4, 5)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cv_dataset(rows: Sequence[dict], expected_folds: int) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "image",
        "probe_target",
        "probe_label",
        "probe_group_id",
        "probe_image_sha256",
        "probe_cv_fold",
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"CV row {index} is missing fields: {sorted(missing)}")

    image_paths = [str(row["image"]) for row in rows]
    image_hashes = [str(row["probe_image_sha256"]) for row in rows]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("CV input must contain at most one row per image path")
    if len(image_hashes) != len(set(image_hashes)):
        raise ValueError("CV input must contain at most one row per image-content hash")

    targets = np.asarray([int(row["probe_label"]) for row in rows], dtype=np.int64)
    target_values = np.asarray([float(row["probe_target"]) for row in rows])
    if not np.array_equal(targets, target_values.astype(np.int64)):
        raise ValueError("probe_label and probe_target disagree")
    labels = np.asarray(sorted(set(targets.tolist())), dtype=np.int64)
    if not np.array_equal(labels, np.asarray(DEFAULT_LABELS)):
        raise ValueError(f"Expected Count labels {DEFAULT_LABELS}, found {labels.tolist()}")

    folds = np.asarray([int(row["probe_cv_fold"]) for row in rows], dtype=np.int64)
    expected = np.arange(expected_folds, dtype=np.int64)
    if not np.array_equal(np.unique(folds), expected):
        raise ValueError(f"Expected folds {expected.tolist()}, found {np.unique(folds).tolist()}")
    for fold in expected:
        fold_labels = set(targets[folds == fold].tolist())
        if fold_labels != set(DEFAULT_LABELS):
            raise ValueError(f"Fold {fold} does not contain every label: {sorted(fold_labels)}")
    return targets, folds


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] = DEFAULT_LABELS,
) -> dict:
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if true.shape != predicted.shape:
        raise ValueError(f"Prediction shape {predicted.shape} does not match targets {true.shape}")

    per_label_accuracy = {}
    for label in labels:
        mask = true == label
        if not mask.any():
            raise ValueError(f"Cannot compute macro accuracy: label {label} is absent")
        per_label_accuracy[str(label)] = float(np.mean(predicted[mask] == label))
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_accuracy": float(np.mean(list(per_label_accuracy.values()))),
        "macro_f1": float(f1_score(true, predicted, labels=list(labels), average="macro", zero_division=0)),
        "per_label_accuracy": per_label_accuracy,
        "confusion_matrix": confusion_matrix(true, predicted, labels=list(labels)).tolist(),
    }


def stratified_bootstrap_macro_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[int] = DEFAULT_LABELS,
    samples: int = 2000,
) -> dict:
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    indices_by_label = {}
    for label in labels:
        indices = np.flatnonzero(true == label)
        if len(indices) == 0:
            raise ValueError(f"Cannot bootstrap absent label {label}")
        indices_by_label[label] = indices

    estimates = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        label_accuracies = []
        for label, indices in indices_by_label.items():
            sampled = np.random.choice(indices, size=len(indices), replace=True)
            label_accuracies.append(np.mean(predicted[sampled] == label))
        estimates[sample_index] = np.mean(label_accuracies)

    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "method": "label-stratified image bootstrap",
        "samples": samples,
        "lower": float(lower),
        "upper": float(upper),
        "standard_error": float(np.std(estimates, ddof=1)),
    }


def cross_validated_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, list[dict]]:
    if c_value <= 0:
        raise ValueError("C must be positive")
    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    folds = np.asarray(folds, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != len(targets) or len(targets) != len(folds):
        raise ValueError("Features, targets, and fold ids have incompatible shapes")
    if not np.isfinite(matrix).all():
        raise ValueError("Probe features contain non-finite values")

    predictions = np.full(len(targets), -1, dtype=np.int64)
    fold_results = []
    for fold in sorted(set(folds.tolist())):
        test_mask = folds == fold
        train_mask = ~test_mask
        train_labels = set(targets[train_mask].tolist())
        if train_labels != set(DEFAULT_LABELS):
            raise ValueError(f"Training partition for fold {fold} lacks labels: {train_labels}")

        scaler = StandardScaler()
        train_features = scaler.fit_transform(matrix[train_mask])
        test_features = scaler.transform(matrix[test_mask])
        classifier = LogisticRegression(
            C=c_value,
            l1_ratio=0.0,
            class_weight="balanced",
            solver="lbfgs",
            fit_intercept=True,
            max_iter=max_iter,
            tol=tolerance,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(train_features, targets[train_mask])
        fold_predictions = classifier.predict(test_features).astype(np.int64)
        predictions[test_mask] = fold_predictions
        convergence_warnings = [
            str(warning.message) for warning in caught if issubclass(warning.category, ConvergenceWarning)
        ]
        fold_results.append(
            {
                "fold": int(fold),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "metrics": classification_metrics(targets[test_mask], fold_predictions),
                "iterations": int(np.max(classifier.n_iter_)),
                "converged": not convergence_warnings,
                "convergence_warnings": convergence_warnings,
            }
        )

    if (predictions < 0).any():
        raise AssertionError("At least one image did not receive an out-of-fold prediction")
    return predictions, fold_results


def evaluate_condition(
    representations: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    *,
    condition: str,
    c_value: float,
    max_iter: int,
    tolerance: float,
    bootstrap_samples: int,
) -> tuple[dict, np.ndarray]:
    representations = np.asarray(representations, dtype=np.float32)
    if representations.ndim != 3:
        raise ValueError("Representations must have shape [examples, hidden states, dimensions]")
    num_hidden_states = representations.shape[1]
    predictions = np.empty((num_hidden_states, len(targets)), dtype=np.int64)
    layer_results = []

    for hidden_state_index in range(num_hidden_states):
        layer_predictions, fold_results = cross_validated_predictions(
            representations[:, hidden_state_index, :],
            targets,
            folds,
            c_value=c_value,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        predictions[hidden_state_index] = layer_predictions
        fold_accuracies = [fold["metrics"]["macro_accuracy"] for fold in fold_results]
        layer_results.append(
            {
                "hidden_state_index": hidden_state_index,
                "oof": classification_metrics(targets, layer_predictions),
                "macro_accuracy_95ci": stratified_bootstrap_macro_accuracy(
                    targets,
                    layer_predictions,
                    samples=bootstrap_samples,
                ),
                "fold_macro_accuracy_mean": float(np.mean(fold_accuracies)),
                "fold_macro_accuracy_std": float(np.std(fold_accuracies, ddof=1)),
                "fold_results": fold_results,
            }
        )

    return {
        "condition": condition,
        "num_hidden_states": num_hidden_states,
        "layer_results": layer_results,
    }, predictions


def plot_results(results: dict[str, dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"image": "#729CFE", "no_image": "#88B06D", "shuffled_image": "#F5A83D"}
    labels = {"image": "Image", "no_image": "No image", "shuffled_image": "Shuffled image"}
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for condition, result in results.items():
        layers = [row["hidden_state_index"] for row in result["layer_results"]]
        accuracy = [100.0 * row["oof"]["macro_accuracy"] for row in result["layer_results"]]
        lower = [100.0 * row["macro_accuracy_95ci"]["lower"] for row in result["layer_results"]]
        upper = [100.0 * row["macro_accuracy_95ci"]["upper"] for row in result["layer_results"]]
        axis.plot(layers, accuracy, marker="o", markersize=3, label=labels[condition], color=colors[condition])
        axis.fill_between(layers, lower, upper, color=colors[condition], alpha=0.14, linewidth=0)

    axis.set(
        title="Image-unique 5-fold Count linear probe",
        xlabel="Hidden-state index (0 = embedding output)",
        ylabel="Out-of-fold macro accuracy (%)",
        ylim=(0, 100),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--data-file",
        default="dataset/probing/count_cv_unique/count_cv_unique_all.jsonl",
    )
    parser.add_argument("--image-root", default="dataset")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Fixed inverse L2 regularization strength; smaller C means stronger regularization",
    )
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--no-image-batch-size",
        type=int,
        default=32,
        help="Fixed batch size for a shared generic no-image prompt",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-cache", action="store_true")
    add_eval_result_args(parser, default_category="find_layers", default_output_name="results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.c <= 0:
        raise ValueError("--c must be positive")
    run = prepare_eval_run_from_args(
        args,
        default_category="find_layers",
        default_output_name="results.json",
    )

    configure_run_logging(
        run,
        kind="count-last-token-linear-probe-cv",
        log_format="%(asctime)s | %(levelname)s | %(message)s",
    )
    setup_seed(args.seed)

    data_path = Path(args.data_file)
    rows = load_jsonl(data_path)
    targets, folds = validate_cv_dataset(rows, args.folds)
    cache_dir = run.artifact_dir("representations")
    model, processor = load_model_and_processor(
        args.model_name_or_path,
        use_cache=False,
        padding_side="left",
        attn_implementation=args.attn_implementation,
    )

    for condition in args.conditions:
        cache_path = cache_dir / f"{condition}_all.npz"
        if cache_path.exists() and not args.overwrite_cache:
            logger.info("Using cached representations: %s", cache_path)
        else:
            fixed_no_image_prompt = condition == "no_image" and len({row["question"] for row in rows}) == 1
            extraction_batch_size = args.no_image_batch_size if fixed_no_image_prompt else args.batch_size
            extract_representations(
                model=model,
                processor=processor,
                rows=rows,
                split="all",
                condition=condition,
                output_path=cache_path,
                image_root=args.image_root,
                batch_size=extraction_batch_size,
                max_length=args.max_length,
                num_workers=args.num_workers,
                shuffle_group_key="probe_cv_fold",
                pad_to_full_batch=fixed_no_image_prompt,
            )
        validate_representation_cache(cache_path, condition)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    condition_results = {}
    prediction_arrays = {}
    for condition in args.conditions:
        cache = load_cache(cache_dir / f"{condition}_all.npz")
        cache_targets = cache["targets"].reshape(-1).astype(np.int64)
        if not np.array_equal(cache_targets, targets):
            raise RuntimeError(f"Cached target order does not match {data_path} for {condition}")
        cached_images = cache["images"].astype(str).tolist()
        expected_images = [str(row["image"]) for row in rows]
        if cached_images != expected_images:
            raise RuntimeError(f"Cached image order does not match {data_path} for {condition}")
        condition_result, predictions = evaluate_condition(
            cache["representations"],
            targets,
            folds,
            condition=condition,
            c_value=args.c,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
            bootstrap_samples=args.bootstrap_samples,
        )
        condition_results[condition] = condition_result
        prediction_arrays[f"{condition}_predictions"] = predictions

    prediction_path = run.artifact_path("oof_predictions.npz")
    np.savez_compressed(
        prediction_path,
        targets=targets,
        folds=folds,
        image_sha256=np.asarray([row["probe_image_sha256"] for row in rows]),
        **prediction_arrays,
    )
    plot_path = run.artifact_path("layerwise_cv_accuracy.png")
    plot_results(condition_results, plot_path)

    result = {
        "model_name_or_path": args.model_name_or_path,
        "dataset": {
            "path": str(data_path),
            "sha256": file_sha256(data_path),
            "rows": len(rows),
            "unique_images": len({row["probe_image_sha256"] for row in rows}),
            "folds": args.folds,
            "labels": list(DEFAULT_LABELS),
        },
        "probe": {
            "type": "multinomial logistic linear probe",
            "standardization": "StandardScaler fit independently on each training fold",
            "regularization": "L2",
            "c": args.c,
            "c_definition": "inverse L2 regularization strength; smaller C is stronger",
            "class_weight": "balanced",
            "solver": "lbfgs",
            "fit_intercept": True,
            "max_iter": args.max_iter,
            "tolerance": args.tolerance,
            "hyperparameter_selection": "none; configuration fixed before evaluation",
            "layer_selection": "none; every hidden-state index is reported",
        },
        "evaluation": {
            "protocol": "image-unique stratified five-fold cross-validation",
            "primary_metric": "pooled out-of-fold macro accuracy",
            "confidence_interval": "label-stratified image bootstrap, 95% percentile interval",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "feature_extraction": {
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "num_workers": args.num_workers,
            "attn_implementation": args.attn_implementation,
            "precision": "bfloat16 model forward; float16 cache; float32 probe fitting",
            "shuffled_image_policy": "label-agnostic derangement performed independently within each fold",
        },
        "conditions": condition_results,
        "oof_predictions": str(prediction_path),
        "plot": str(plot_path),
    }
    result_path = write_json_result(run, result, kind="count-last-token-linear-probe-cv")
    logger.info("Results saved to %s", result_path)


if __name__ == "__main__":
    main()
