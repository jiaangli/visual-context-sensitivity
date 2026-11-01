"""Select a single intervention layer from a saved layer-search trace."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

DEFAULT_ONSET_THRESHOLD = 0.05
DEFAULT_ONSET_PERSISTENCE = 2
DIFFERENCE_ONSET_METHOD = "first_sustained_nondecreasing_probability_gap_effect"


def _mean_vector(
    condition: Mapping[str, Any],
    answer: str,
    *,
    condition_name: str,
) -> list[float]:
    try:
        values = condition[answer]["mean"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{condition_name}.{answer}.mean is missing from probability-gap data") from error
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{condition_name}.{answer}.mean must be a sequence of numbers")
    vector = [float(value) for value in values]
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{condition_name}.{answer}.mean must contain finite layer values")
    return vector


def probability_gap_effect_curve(
    probability_gap_plot: Mapping[str, Any],
) -> list[float]:
    """Return the per-layer causal change in the source-vs-target gap.

    For each layer this computes::

        (P_patched(source) - P_patched(target))
        - (P_baseline(source) - P_baseline(target))

    A positive value is movement in the intended patching direction.
    """
    try:
        baseline = probability_gap_plot["baseline"]
        patched = probability_gap_plot["patched"]
    except (KeyError, TypeError) as error:
        raise ValueError("probability-gap data must contain baseline and patched conditions") from error

    baseline_source = _mean_vector(baseline, "source", condition_name="baseline")
    baseline_target = _mean_vector(baseline, "target", condition_name="baseline")
    patched_source = _mean_vector(patched, "source", condition_name="patched")
    patched_target = _mean_vector(patched, "target", condition_name="patched")
    lengths = {
        len(baseline_source),
        len(baseline_target),
        len(patched_source),
        len(patched_target),
    }
    if len(lengths) != 1:
        raise ValueError("probability-gap mean vectors must have equal lengths")

    return [
        (patched_source[layer] - patched_target[layer]) - (baseline_source[layer] - baseline_target[layer])
        for layer in range(len(baseline_source))
    ]


def select_difference_onset(
    search_trace: Mapping[str, Any],
    *,
    threshold: float = DEFAULT_ONSET_THRESHOLD,
    persistence: int = DEFAULT_ONSET_PERSISTENCE,
) -> dict[str, Any]:
    """Find the first sustained, material divergence in a probability-gap trace.

    The selected layer is always one of ``selected_layers``. The material-effect
    cutoff is an absolute change in answer probability. Requiring consecutive
    crossings avoids selecting a noisy one-layer spike. The confirming effect
    must also be nondecreasing, so a layer at the top of an already weakening
    spike is not labeled as its onset. If no qualifying onset exists inside the
    important range, the range's final layer is retained as a conservative
    fallback instead of selecting a downstream propagation layer.
    """
    if not 0 < threshold <= 2:
        raise ValueError(f"threshold must be in (0, 2], got {threshold}")
    if isinstance(persistence, bool) or not isinstance(persistence, int):
        raise TypeError("persistence must be a positive integer")
    if persistence <= 0:
        raise ValueError(f"persistence must be positive, got {persistence}")

    try:
        plot_data = search_trace["probability_gap_plot"]
    except (KeyError, TypeError) as error:
        raise ValueError("search trace does not contain probability_gap_plot data") from error
    if not isinstance(plot_data, Mapping):
        raise TypeError("probability_gap_plot must be a mapping")

    selected_layers = search_trace.get("selected_layers")
    plotted_patch_layers = plot_data.get("patch_layers")
    patch_layers = selected_layers or plotted_patch_layers
    if not isinstance(patch_layers, Sequence) or isinstance(patch_layers, (str, bytes)):
        raise TypeError("search trace patch_layers must be a non-empty sequence")
    patch_layers = [int(layer) for layer in patch_layers]
    if not patch_layers:
        raise ValueError("search trace does not contain a non-empty patch layer list")
    if patch_layers != sorted(set(patch_layers)):
        raise ValueError("patch layers must be strictly increasing and unique")
    if selected_layers and plotted_patch_layers:
        plotted_patch_layers = [int(layer) for layer in plotted_patch_layers]
        if plotted_patch_layers != patch_layers:
            raise ValueError("probability-gap patch layers do not match selected_layers")

    effect_curve = probability_gap_effect_curve(plot_data)
    search_start = min(patch_layers)
    search_end = max(patch_layers)
    if search_start < 0 or search_end >= len(effect_curve):
        raise ValueError(f"patch layers must be within [0, {len(effect_curve) - 1}]")

    downstream_effect = effect_curve[search_start:]
    peak_offset = max(range(len(downstream_effect)), key=downstream_effect.__getitem__)
    peak_layer = search_start + peak_offset
    peak_effect = downstream_effect[peak_offset]
    if peak_effect <= 0:
        raise ValueError("patched source-vs-target probability gap has no positive downstream effect")

    onset_layer = next(
        (
            layer
            for layer in patch_layers
            if layer + persistence <= len(effect_curve) and layer + persistence - 1 <= search_end
            if all(effect_curve[candidate] >= threshold for candidate in range(layer, layer + persistence))
            and all(
                effect_curve[candidate + 1] >= effect_curve[candidate]
                for candidate in range(layer, layer + persistence - 1)
            )
        ),
        None,
    )
    used_fallback = onset_layer is None
    if onset_layer is None:
        onset_layer = search_end

    return {
        "schema_version": 2,
        "method": DIFFERENCE_ONSET_METHOD,
        "metric": (
            "(patched_source_probability-patched_target_probability)-"
            "(baseline_source_probability-baseline_target_probability)"
        ),
        "layer": onset_layer,
        "effect_at_layer": effect_curve[onset_layer],
        "threshold": threshold,
        "persistence": persistence,
        "confirmation_trend": "nondecreasing",
        "confirmation_layers": ([] if used_fallback else list(range(onset_layer, onset_layer + persistence))),
        "confirmation_effects": ([] if used_fallback else effect_curve[onset_layer : onset_layer + persistence]),
        "used_fallback": used_fallback,
        "fallback_reason": (
            "no_sustained_nondecreasing_threshold_crossing_within_selected_layers" if used_fallback else None
        ),
        "search_start_layer": search_start,
        "search_end_layer": search_end,
        "downstream_peak_search_end_layer": len(effect_curve) - 1,
        "peak_layer": peak_layer,
        "peak_effect": peak_effect,
    }


def annotate_difference_onset(
    search_trace: MutableMapping[str, Any],
    *,
    threshold: float = DEFAULT_ONSET_THRESHOLD,
    persistence: int = DEFAULT_ONSET_PERSISTENCE,
) -> dict[str, Any]:
    """Compute and store the recommended single-layer target in ``search_trace``."""
    selection = select_difference_onset(
        search_trace,
        threshold=threshold,
        persistence=persistence,
    )
    search_trace["difference_onset"] = selection
    search_trace["target_layer"] = selection["layer"]
    return selection


def _require_selected_target(search_trace: Mapping[str, Any], target_layer: int) -> int:
    selected_layers = search_trace.get("selected_layers")
    if not isinstance(selected_layers, Sequence) or isinstance(selected_layers, (str, bytes)):
        raise TypeError("search trace selected_layers must be a sequence")
    normalized_layers = [int(layer) for layer in selected_layers]
    if target_layer not in normalized_layers:
        raise ValueError(f"target layer {target_layer} is outside selected_layers {normalized_layers}")
    return target_layer


def resolve_difference_onset(
    search_trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Read a stored onset selection, or derive one from the saved probability-gap data."""
    selection = search_trace.get("difference_onset")
    if (
        isinstance(selection, Mapping)
        and selection.get("layer") is not None
        and selection.get("method") == DIFFERENCE_ONSET_METHOD
    ):
        resolved = dict(selection)
        resolved["layer"] = _require_selected_target(search_trace, int(resolved["layer"]))
        return resolved

    parameters = search_trace.get("parameters", {})
    if not isinstance(parameters, Mapping):
        parameters = {}
    return select_difference_onset(
        search_trace,
        threshold=float(parameters.get("onset_threshold", DEFAULT_ONSET_THRESHOLD)),
        persistence=int(parameters.get("onset_persistence", DEFAULT_ONSET_PERSISTENCE)),
    )


def resolve_target_layer(search_trace: Mapping[str, Any]) -> int:
    """Read the stored target layer, or derive the onset from the saved probability-gap data."""
    stored_target = search_trace.get("target_layer")
    if stored_target is not None:
        _require_selected_target(search_trace, int(stored_target))
    if "probability_gap_plot" in search_trace:
        return _require_selected_target(search_trace, int(resolve_difference_onset(search_trace)["layer"]))
    if stored_target is not None:
        return int(stored_target)
    raise ValueError("search trace has neither probability-gap data nor target_layer")
