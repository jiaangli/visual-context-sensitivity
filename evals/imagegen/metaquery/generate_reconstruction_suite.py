from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import safetensors.torch
import torch
from PIL import Image
from tqdm import tqdm

from src.imagegen.metaquery.metaquery import MetaQuery, MetaQueryConfig
from src.imagegen.metaquery.reconstruction_data import (
    TEST_SET_NAMES,
    color_correct,
    load_test_suite,
    records_to_jsonl,
    stable_image_seed,
)
from src.imagegen.metaquery.trainer_utils import find_newest_checkpoint

LOGGER = logging.getLogger(__name__)


def _checkpoint_state_for_current_transformers(state: dict[str, torch.Tensor], mllm_id: str) -> dict[str, torch.Tensor]:
    """Translate checkpoints written before the Qwen2.5-VL module rename."""
    if "Qwen2.5-VL" not in mllm_id:
        return state

    # Transformers moved the vision tower below ``model`` and renamed the
    # language module from ``model`` to ``model.language_model``. Without
    # this translation all 824 backbone tensors remain silently untrained.
    old_visual = "model.mllm_backbone.visual."
    new_visual = "model.mllm_backbone.model.visual."
    old_language = "model.mllm_backbone.model."
    new_language = "model.mllm_backbone.model.language_model."
    translated: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith(old_visual):
            key = new_visual + key.removeprefix(old_visual)
        elif key.startswith(old_language) and not key.startswith(new_language):
            key = new_language + key.removeprefix(old_language)
        if key in translated:
            raise RuntimeError(f"checkpoint key translation collision: {key}")
        translated[key] = value
    return translated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-root", default="dataset/whatifvis")
    parser.add_argument("--tasks", nargs="+", choices=TEST_SET_NAMES, default=list(TEST_SET_NAMES))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-correct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-images-per-task",
        type=int,
        default=0,
        help="Testing aid; zero evaluates every unique image.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
    temporary.replace(path)


def _save_image_atomic(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}")
    image.save(temporary)
    temporary.replace(destination)


def _manifest(args: argparse.Namespace, checkpoint_path: str, status: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "command": list(sys.argv),
        "hostname": socket.gethostname(),
        "slurm": {
            key.lower(): value
            for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
            if (value := os.environ.get(key))
        },
        "model_alias": args.model_alias,
        "checkpoint_path": checkpoint_path,
        "dataset_root": str(Path(args.dataset_root)),
        "tasks": list(args.tasks),
        "generation": {
            "batch_size": args.batch_size,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "image_guidance_scale": args.image_guidance_scale,
            "seed": args.seed,
            "color_correct": args.color_correct,
            "deterministic_seed_scope": "per task and normalized input path",
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MetaQuery reconstruction generation")

    suite_dir = Path(args.suite_dir)
    model_dir = suite_dir / "models" / args.model_alias
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "generation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )

    checkpoint_path = find_newest_checkpoint(args.checkpoint_path)
    if checkpoint_path is None:
        raise FileNotFoundError(f"no checkpoint found below {args.checkpoint_path}")
    checkpoint_path = str(Path(checkpoint_path).resolve())
    manifest_path = model_dir / "generation.manifest.json"
    manifest = _manifest(args, checkpoint_path, "running")
    _write_json(manifest_path, manifest)

    try:
        suite = load_test_suite(args.dataset_root, args.tasks)
        if args.max_images_per_task > 0:
            suite = {task: records[: args.max_images_per_task] for task, records in suite.items()}

        LOGGER.info("Loading MetaQuery checkpoint %s", checkpoint_path)
        config = MetaQueryConfig.from_pretrained(checkpoint_path)
        with torch.device("cpu"):
            model = MetaQuery(config)
            checkpoint_state = _checkpoint_state_for_current_transformers(
                safetensors.torch.load_file(
                    str(Path(checkpoint_path) / "model.safetensors"),
                    device="cpu",
                ),
                config.mllm_id,
            )
            incompatible = model.load_state_dict(checkpoint_state, strict=False)
        LOGGER.info(
            "Checkpoint loaded: %d missing keys, %d unexpected keys",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "checkpoint is incompatible with the instantiated model: "
                f"missing={incompatible.missing_keys[:20]}, "
                f"unexpected={incompatible.unexpected_keys[:20]}"
            )
        pipeline = model.to(device="cuda", dtype=torch.bfloat16).eval()
        for parameter in pipeline.parameters():
            parameter.requires_grad_(False)

        dataset_summaries: dict[str, dict[str, object]] = {}
        for task, records in suite.items():
            task_dir = model_dir / task
            images_dir = task_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            record_rows: list[dict[str, object]] = []
            generated = 0
            reused = 0

            for start in tqdm(
                range(0, len(records), args.batch_size),
                desc=f"{args.model_alias}/{task}",
            ):
                batch_records = records[start : start + args.batch_size]
                pending = []
                for record in batch_records:
                    destination = images_dir / record.output_name
                    if args.resume and destination.is_file() and destination.stat().st_size > 0:
                        reused += 1
                    else:
                        pending.append(record)

                if pending:
                    input_images = [
                        Image.open(record.input_path).convert("RGB").resize((512, 512)) for record in pending
                    ]
                    generators = [
                        torch.Generator(device="cuda").manual_seed(
                            stable_image_seed(args.seed, task, record.input_path)
                        )
                        for record in pending
                    ]
                    with torch.inference_mode():
                        output_images = pipeline.sample_images(
                            input_images=input_images,
                            caption=[""] * len(input_images),
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            image_guidance_scale=args.image_guidance_scale,
                            enable_progress_bar=False,
                            generator=generators,
                        )
                    if len(output_images) != len(pending):
                        raise RuntimeError(f"model returned {len(output_images)} images for a batch of {len(pending)}")
                    for output_image, input_image, record in zip(output_images, input_images, pending):
                        if args.color_correct:
                            output_image = color_correct(output_image, input_image)
                        _save_image_atomic(output_image, images_dir / record.output_name)
                        generated += 1

                for record in batch_records:
                    row = record.to_dict()
                    row["reconstruction_path"] = str((images_dir / record.output_name).resolve())
                    row["image_seed"] = stable_image_seed(args.seed, task, record.input_path)
                    record_rows.append(row)

            missing_outputs = [
                row["reconstruction_path"] for row in record_rows if not Path(str(row["reconstruction_path"])).is_file()
            ]
            if missing_outputs:
                raise RuntimeError(f"{task}: {len(missing_outputs)} expected reconstructions are missing")

            records_path = task_dir / "records.jsonl"
            records_to_jsonl(record_rows, records_path)
            dataset_summaries[task] = {
                "dataset_file": str((Path(args.dataset_root) / f"{task}_test.jsonl").resolve()),
                "num_unique_inputs": len(records),
                "num_generated_this_run": generated,
                "num_reused": reused,
                "images_dir": str(images_dir.resolve()),
                "records_file": str(records_path.resolve()),
            }
            LOGGER.info(
                "%s complete: %d images (%d generated, %d reused)",
                task,
                len(records),
                generated,
                reused,
            )

        result = _manifest(args, checkpoint_path, "complete")
        result["datasets"] = dataset_summaries
        result["total_images"] = sum(int(value["num_unique_inputs"]) for value in dataset_summaries.values())
        _write_json(model_dir / "generation.json", result)
        _write_json(manifest_path, result)
    except BaseException as exc:
        failed = _manifest(args, checkpoint_path, "failed")
        failed["error"] = repr(exc)
        _write_json(manifest_path, failed)
        raise


if __name__ == "__main__":
    main()
