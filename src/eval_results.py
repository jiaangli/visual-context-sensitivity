from __future__ import annotations

import json
import logging
import os
import re
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
DEFAULT_EVAL_ROOT = "results/eval"
EXPERIMENT_CATEGORIES = (
    "metaquery",
    "sft",
    "find_layers",
    "steering",
)
CATEGORY_ALIASES = {
    "image_recon": "metaquery",
    "probing": "find_layers",
}
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def utc_timestamp(now: Optional[datetime] = None) -> str:
    """Return a compact, sortable UTC timestamp."""
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT)


def normalize_expid(expid: str) -> str:
    """Normalize an experiment id into a safe directory component."""
    normalized = _SAFE_COMPONENT.sub("-", expid.strip()).strip("-._")
    if not normalized:
        raise ValueError("expid must contain at least one letter or number")
    return normalized


def normalize_category(category: str) -> str:
    """Return a canonical experiment category and reject accidental buckets."""
    normalized = normalize_expid(category)
    normalized = CATEGORY_ALIASES.get(normalized, normalized)
    if normalized not in EXPERIMENT_CATEGORIES:
        choices = ", ".join(EXPERIMENT_CATEGORIES)
        raise ValueError(f"unknown experiment category {category!r}; expected one of: {choices}")
    return normalized


def validate_timestamp(timestamp: str) -> str:
    """Validate and return a timestamp in the canonical UTC format."""
    try:
        parsed = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise ValueError(f"timestamp must use UTC format {TIMESTAMP_FORMAT!r}, got {timestamp!r}") from exc
    if parsed.strftime(TIMESTAMP_FORMAT) != timestamp:
        raise ValueError(f"invalid timestamp: {timestamp!r}")
    return timestamp


def _validate_output_name(output_name: str) -> str:
    path = Path(output_name)
    if not output_name or path.name != output_name or output_name in {".", ".."}:
        raise ValueError("output_name must be a filename, not a path")
    return output_name


def _utc_iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EvalRun:
    """Resolved paths and identifiers for one evaluation invocation."""

    category: str
    expid: str
    timestamp: str
    run_dir: Path
    result_path: Path
    log_path: Path
    started_at_utc: str

    @property
    def manifest_path(self) -> Path:
        """Return the provenance manifest paired with this result artifact."""
        return self.run_dir / f"{self.result_path.stem}.manifest.json"

    def artifact_path(self, relative_path: str | Path) -> Path:
        """Return a safe path below this run directory and create its parent."""
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path must stay inside the run directory")
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_dir(self, relative_path: str | Path) -> Path:
        """Return a safe artifact directory below this run directory."""
        path = self.artifact_path(relative_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def metadata(self, *, kind: str, extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "category": self.category,
            "expid": self.expid,
            "timestamp": self.timestamp,
            "timestamp_timezone": "UTC",
            "kind": kind,
            "started_at_utc": self.started_at_utc,
            "command": list(sys.argv),
            "working_directory": str(Path.cwd()),
            "hostname": socket.gethostname(),
        }
        slurm = {
            key.lower(): value
            for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
            if (value := os.environ.get(key))
        }
        if slurm:
            metadata["slurm"] = slurm
        if extra:
            metadata.update(extra)
        return metadata


def prepare_eval_run(
    *,
    category: str,
    expid: str,
    output_root: str | Path = DEFAULT_EVAL_ROOT,
    timestamp: Optional[str] = None,
    output_name: str = "results.json",
) -> EvalRun:
    """Create and return the standardized paths for an evaluation run.

    The result artifact is always written under
    ``<root>/<category>/<expid>/<timestamp>/``.
    """
    normalized_category = normalize_category(category)
    normalized_expid = normalize_expid(expid)
    run_timestamp = validate_timestamp(timestamp) if timestamp else utc_timestamp()
    output_name = _validate_output_name(output_name)

    run_dir = Path(output_root) / normalized_category / normalized_expid / run_timestamp
    result_path = run_dir / output_name

    run_dir.mkdir(parents=True, exist_ok=True)
    return EvalRun(
        category=normalized_category,
        expid=normalized_expid,
        timestamp=run_timestamp,
        run_dir=run_dir,
        result_path=result_path,
        log_path=run_dir / f"{Path(output_name).stem}.log",
        started_at_utc=_utc_iso_timestamp(),
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
    os.replace(temporary_path, path)


def write_run_manifest(
    run: EvalRun,
    *,
    kind: str,
    status: str,
    metadata: Optional[Mapping[str, Any]] = None,
    error: Optional[str | BaseException] = None,
) -> Path:
    """Atomically record run provenance and lifecycle state."""
    if status not in {"running", "complete", "failed"}:
        raise ValueError(f"invalid run status: {status!r}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "updated_at_utc": _utc_iso_timestamp(),
        "run": run.metadata(kind=kind, extra=metadata),
        "artifacts": {
            "result": str(run.result_path),
            "log": str(run.log_path),
        },
    }
    if error is not None:
        payload["error"] = error
    _atomic_write_json(run.manifest_path, payload)
    return run.manifest_path


def configure_run_logging(
    run: EvalRun,
    *,
    kind: str,
    metadata: Optional[Mapping[str, Any]] = None,
    level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> Path:
    """Configure logging and write the initial ``running`` manifest."""
    write_run_manifest(run, kind=kind, status="running", metadata=metadata)
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(), logging.FileHandler(run.log_path, encoding="utf-8")],
        force=True,
    )
    return run.log_path


def mark_run_failed(
    run: EvalRun,
    *,
    kind: str,
    error: BaseException | str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Mark a run failed while retaining its partial artifacts and log."""
    return write_run_manifest(run, kind=kind, status="failed", metadata=metadata, error=str(error))


def mark_run_complete(
    run: EvalRun,
    *,
    kind: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Mark a non-JSON artifact run complete."""
    if not run.result_path.is_file():
        raise FileNotFoundError(f"primary run artifact does not exist: {run.result_path}")
    return write_run_manifest(run, kind=kind, status="complete", metadata=metadata)


def add_eval_result_args(
    parser: Any,
    *,
    default_category: Optional[str],
    default_output_name: str = "results.json",
) -> None:
    """Add the common result-layout arguments to an argparse parser."""
    argparse_timestamp_format = TIMESTAMP_FORMAT.replace("%", "%%")
    parser.add_argument("--expid", required=True, help="Experiment id used to group related eval jobs")
    category_kwargs: dict[str, Any]
    if default_category is None:
        category_kwargs = {"required": True}
        category_default = "required"
    else:
        category_kwargs = {"default": normalize_category(default_category)}
        category_default = normalize_category(default_category)
    parser.add_argument(
        "--eval_category",
        **category_kwargs,
        type=normalize_category,
        choices=EXPERIMENT_CATEGORIES,
        help=("Experiment category: " + ", ".join(EXPERIMENT_CATEGORIES) + f" ({category_default})"),
    )
    parser.add_argument(
        "--output_root",
        default=DEFAULT_EVAL_ROOT,
        help="Root directory for eval artifacts (default: results/eval)",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help=f"Shared run timestamp in UTC format {argparse_timestamp_format}; generated when omitted",
    )
    parser.add_argument(
        "--output_name",
        default=default_output_name,
        help="Result filename inside the expid/timestamp run directory",
    )


def prepare_eval_run_from_args(
    args: Any,
    *,
    default_category: Optional[str],
    default_output_name: str = "results.json",
) -> EvalRun:
    """Resolve an :class:`EvalRun` from argparse/dataclass attributes."""
    category = getattr(args, "eval_category", None) or default_category
    if category is None:
        raise ValueError("eval_category is required")
    return prepare_eval_run(
        category=category,
        expid=args.expid,
        output_root=getattr(args, "output_root", DEFAULT_EVAL_ROOT),
        timestamp=getattr(args, "timestamp", None),
        output_name=getattr(args, "output_name", default_output_name),
    )


def write_json_result(
    run: EvalRun,
    payload: MutableMapping[str, Any],
    *,
    kind: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Attach run metadata and atomically write a JSON result artifact."""
    payload["run"] = run.metadata(kind=kind, extra=metadata)
    _atomic_write_json(run.result_path, payload)
    write_run_manifest(run, kind=kind, status="complete", metadata=metadata)
    return run.result_path
