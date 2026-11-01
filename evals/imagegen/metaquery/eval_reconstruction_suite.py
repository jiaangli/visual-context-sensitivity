from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision.transforms import v2
from tqdm import tqdm

from src.imagegen.metaquery.reconstruction_data import (
    TEST_SET_NAMES,
    ReconstructionRecord,
    count_factual_path,
    is_count_animal,
)

LOGGER = logging.getLogger(__name__)
PREPROCESS = v2.Compose(
    [
        v2.Lambda(lambda image: image.convert("RGB")),
        v2.Resize((512, 512), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(512),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TEST_SET_NAMES, default=list(TEST_SET_NAMES))
    parser.add_argument("--factual-dir", default="dataset/whatifvis/images/vlms_are_biased_factual")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
    temporary.replace(path)


def _load_records(path):
    records = []
    fields = set(ReconstructionRecord.__dataclass_fields__)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record = ReconstructionRecord(**{k: row[k] for k in fields})
            reconstruction = Path(row["reconstruction_path"])
            if not reconstruction.is_file():
                raise FileNotFoundError(f"Reconstruction file not found: {reconstruction}")
            records.append((record, reconstruction))
    return records


class PixelMetricEvaluator:
    """Reusable evaluator matching the repository's reconstruction metrics."""

    def __init__(self, device: str, batch_size: int) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

    def evaluate(self, pairs: list[tuple[Path, Path]], description: str) -> dict[str, int | float | None]:
        self.ssim.reset()
        self.psnr.reset()
        self.lpips.reset()
        processed = 0
        for start in tqdm(range(0, len(pairs), self.batch_size), desc=description):
            batch = pairs[start : start + self.batch_size]
            references = []
            generated = []
            for reference_path, generated_path in batch:
                with Image.open(reference_path) as reference_image:
                    references.append(PREPROCESS(reference_image))
                with Image.open(generated_path) as generated_image:
                    generated.append(PREPROCESS(generated_image))
            reference_tensor = torch.stack(references).to(self.device)
            generated_tensor = torch.stack(generated).to(self.device)
            self.ssim.update(generated_tensor, reference_tensor)
            self.psnr.update(generated_tensor, reference_tensor)
            self.lpips.update(generated_tensor, reference_tensor)
            processed += len(batch)
        if processed == 0:
            return {"num_images": 0, "psnr": None, "ssim": None, "lpips": None}
        return {
            "num_images": processed,
            "psnr": float(self.psnr.compute()),
            "ssim": float(self.ssim.compute()),
            "lpips": float(self.lpips.compute()),
        }


def _comparison(
    counterfactual: dict[str, int | float | None],
    factual: dict[str, int | float | None],
) -> dict[str, float | bool]:
    result: dict[str, float | bool] = {}
    for metric in ("psnr", "ssim", "lpips"):
        counterfactual_value = float(counterfactual[metric])
        factual_value = float(factual[metric])
        result[f"{metric}_counterfactual_minus_factual"] = counterfactual_value - factual_value
        result[f"{metric}_prefers_counterfactual"] = (
            counterfactual_value < factual_value if metric == "lpips" else counterfactual_value > factual_value
        )
    return result


def _metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_alias, model_result in payload["models"].items():
        for task, values in model_result["test_sets"].items():
            rows.append(
                {
                    "model": model_alias,
                    "scope": "five_test",
                    "task": task,
                    "reference": "input",
                    **values,
                }
            )
        for reference in ("counterfactual", "factual"):
            rows.append(
                {
                    "model": model_alias,
                    "scope": "count_animals",
                    "task": "count",
                    "reference": reference,
                    **model_result["count_animals"][reference],
                }
            )
    return rows


def _write_tables(metrics_dir: Path, payload: dict[str, Any]) -> None:
    columns = [
        "model",
        "scope",
        "task",
        "reference",
        "num_images",
        "psnr",
        "ssim",
        "lpips",
    ]
    rows = _metric_rows(payload)
    with (metrics_dir / "metrics.tsv").open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")

    with (metrics_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Five-test reconstruction metrics\n\n")
        handle.write(
            "Reconstructions use Lab mean/std color processing from their input images.\n"
            "All metric inputs are resized to 512x512. Higher PSNR/SSIM and lower LPIPS\n"
            "are better.\n\n"
        )
        handle.write("| Model | Scope | Task | Reference | N | PSNR | SSIM | LPIPS |\n")
        handle.write("|---|---|---|---|---|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['model']} | {row['scope']} | {row['task']} | "
                f"{row['reference']} | {row['num_images']} | "
                f"{float(row['psnr']):.4f} | {float(row['ssim']):.4f} | "
                f"{float(row['lpips']):.4f} |\n"
            )


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    metrics_dir = suite_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(metrics_dir / "evaluation.log", encoding="utf-8"),
        ],
        force=True,
    )

    evaluator = PixelMetricEvaluator(args.device, args.batch_size)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "suite_dir": str(suite_dir.resolve()),
        "protocol": {
            "resolution": "512x512 RGB",
            "color_process": (
                "CIELAB per-channel mean/std matching to the input image, applied during reconstruction generation"
            ),
        },
        "count_factual_note": (
            "The reconstruction color process uses its counterfactual input; it is "
            "not re-fit to the factual reference before factual comparison."
        ),
        "metric_definitions": {
            "psnr": "TorchMetrics PeakSignalNoiseRatio; data range [0,1]",
            "ssim": "TorchMetrics StructuralSimilarityIndexMeasure; data range [0,1]",
            "lpips": "TorchMetrics LPIPS/AlexNet; normalize=True",
        },
        "models": {},
    }

    for model_alias in args.models:
        model_dir = suite_dir / "models" / model_alias
        model_result: dict[str, Any] = {"test_sets": {}, "count_animals": {}}
        count_records: list[tuple[ReconstructionRecord, Path]] = []
        for task in args.tasks:
            records = _load_records(model_dir / task / "records.jsonl")
            if task == "count":
                count_records = records
            pairs = [(Path(record.input_path), reconstruction) for record, reconstruction in records]
            model_result["test_sets"][task] = evaluator.evaluate(pairs, f"metrics/{model_alias}/{task}")

        animals = [(record, reconstruction) for record, reconstruction in count_records if is_count_animal(record)]
        counterfactual = evaluator.evaluate(
            [(Path(record.input_path), reconstruction) for record, reconstruction in animals],
            f"metrics/{model_alias}/count-animals/counterfactual",
        )
        factual = evaluator.evaluate(
            [(count_factual_path(record, args.factual_dir), reconstruction) for record, reconstruction in animals],
            f"metrics/{model_alias}/count-animals/factual",
        )
        model_result["count_animals"] = {
            "counterfactual": counterfactual,
            "factual": factual,
            "comparison": _comparison(counterfactual, factual),
        }
        payload["models"][model_alias] = model_result
        _write_json(metrics_dir / "metrics.partial.json", payload)

    _write_json(metrics_dir / "metrics.json", payload)
    _write_tables(metrics_dir, payload)
    LOGGER.info("Evaluation complete: %s", metrics_dir / "metrics.json")


if __name__ == "__main__":
    main()
