from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.interp.patching_plots import plot_layer_search_trace


def load_search_trace(path: str | Path) -> dict[str, Any]:
    """Load either a standalone trace artifact or a full find-range result."""
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    trace = payload.get("search_trace", payload)
    if not isinstance(trace, dict) or not isinstance(trace.get("evaluations"), list):
        raise ValueError(f"{input_path} does not contain a layer-search trace")
    if not trace["evaluations"]:
        raise ValueError(f"{input_path} contains an empty layer-search trace")
    return trace


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot saved layer-search values without loading a model or dataset.")
    parser.add_argument(
        "search_result",
        help="Standalone trace.json or find_range result JSON containing search_trace",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output .png/.pdf/.html path (default: <input_stem>_plot.png)",
    )
    parser.add_argument(
        "--hide-onset",
        action="store_true",
        help="Omit the selected target-layer vertical line and annotation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    input_path = Path(args.search_result)
    output_path = (
        Path(args.output_file) if args.output_file is not None else input_path.with_name(f"{input_path.stem}_plot.png")
    )
    trace = load_search_trace(input_path)
    plot_layer_search_trace(
        trace,
        output_path=str(output_path),
        show_onset=not args.hide_onset,
    )
    print(f"Saved search-trace plot to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
