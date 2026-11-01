from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.interp.plot_search_trace import load_search_trace
from src.interp.layer_selection import (
    DEFAULT_ONSET_PERSISTENCE,
    DEFAULT_ONSET_THRESHOLD,
    resolve_difference_onset,
    select_difference_onset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the first sustained, nondecreasing probability-gap "
            "Divergence from a saved important-layer search trace."
        )
    )
    parser.add_argument(
        "search_result",
        help="Standalone trace.json or find_range result JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            f"Minimum probability-gap effect required at onset (default: trace value or {DEFAULT_ONSET_THRESHOLD:g})"
        ),
    )
    parser.add_argument(
        "--persistence",
        type=int,
        default=None,
        help=(f"Required consecutive crossings (default: trace value or {DEFAULT_ONSET_PERSISTENCE})"),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete auditable selection record instead of only the layer",
    )
    return parser.parse_args(argv)


def select_from_path(
    path: str | Path,
    *,
    threshold: float | None = None,
    persistence: int | None = None,
) -> dict[str, Any]:
    trace = load_search_trace(path)
    if threshold is None and persistence is None:
        return resolve_difference_onset(trace)
    parameters = trace.get("parameters", {})
    resolved_threshold = (
        threshold if threshold is not None else float(parameters.get("onset_threshold", DEFAULT_ONSET_THRESHOLD))
    )
    required_persistence = (
        persistence if persistence is not None else int(parameters.get("onset_persistence", DEFAULT_ONSET_PERSISTENCE))
    )
    return select_difference_onset(
        trace,
        threshold=resolved_threshold,
        persistence=required_persistence,
    )


def main():
    args = parse_args()
    selection = select_from_path(args.search_result, threshold=args.threshold, persistence=args.persistence)
    if args.json:
        print(json.dumps(selection, indent=2))
    else:
        print(selection["layer"])


if __name__ == "__main__":
    main()
