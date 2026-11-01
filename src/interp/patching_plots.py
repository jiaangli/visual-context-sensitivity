"""
Patching comparison plots for mechanistic interpretability.

Three plot types that clearly communicate whether activation patching works:

1. ``plot_overlay`` — Before/after overlay on a single axis.
2. ``plot_delta_bar`` — Summary bar chart of effect size.
3. ``plot_grid_comparison`` — 2x2 grid comparing two directions — baseline/patched.

All functions accept the ``(mean, std, median)`` probability tuples that
``merge_results()`` already produces (shape ``[num_layers, 2]`` per tensor).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

import plotly.graph_objects as go
import plotly.subplots as sp
import torch
from torch import Tensor

from src.interp.layer_selection import resolve_difference_onset

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Paper palette: blue for source answer, red for target answer
_SRC_HEX = "#5D8DFD"
_TGT_HEX = "#F86762"

_FONT_FAMILY = "CMU Serif"


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.removeprefix("#")
    r, g, b = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _format_layers(layers: List[int]) -> str:
    """Compress [14,15,16,17,18] -> '14–18'."""
    if not layers:
        return ""
    ranges, start = [], layers[0]
    for i in range(1, len(layers)):
        if layers[i] != layers[i - 1] + 1:
            ranges.append(f"{start}–{layers[i - 1]}" if start != layers[i - 1] else str(start))
            start = layers[i]
    ranges.append(f"{start}–{layers[-1]}" if start != layers[-1] else str(start))
    return ", ".join(ranges)


def _hypothesis_title(direction: str, site_label: str, layers: List[int]) -> str:
    """Generate a hypothesis-stating title."""
    layer_str = _format_layers(layers)
    direction_map = {
        "prior_to_context": f"Can prior activations override context? (Patch {site_label} at L{layer_str})",
        "context_to_prior": f"Can context activations override prior? (Patch {site_label} at L{layer_str})",
        "prior_only": f"Do prior activations transfer across samples? (Patch {site_label} at L{layer_str})",
        "context_only": f"Do context activations transfer across samples? (Patch {site_label} at L{layer_str})",
    }
    return direction_map.get(direction, f"Patching {site_label} at L{layer_str}")


def _direction_labels(direction: str) -> Tuple[str, str]:
    """Return (source_label, target_label) for a given direction."""
    labels = {
        "prior_to_context": ("PRI", "CTX"),
        "context_to_prior": ("CTX", "PRI"),
        "prior_only": ("SRC PRI", "TGT PRI"),
        "context_only": ("SRC CTX", "TGT CTX"),
    }
    return labels.get(direction, ("SRC", "TGT"))


def _add_prob_lines(
    fig,
    probs: Tuple[Tensor, Tensor, Tensor],
    n_layers: int,
    color_hex: str,
    label: str,
    dash: str = "solid",
    row: int = None,
    col: int = None,
    show_band: bool = True,
    legend_group: str = "",
):
    """Add a mean line + std band + median dashed line for one answer type."""
    mean, std, median = probs
    x = list(range(n_layers))
    # Only pass row/col when using subplots (not a plain go.Figure)
    rc = dict(row=row, col=col) if row is not None else {}

    if show_band and dash == "solid":
        upper = (mean + std).clamp(0, 1)
        lower = (mean - std).clamp(0, 1)
        fig.add_trace(
            go.Scatter(
                x=x, y=upper.tolist(), mode="lines", line=dict(width=0), showlegend=False, legendgroup=legend_group
            ),
            **rc,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower.tolist(),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_rgba(color_hex, 0.15),
                showlegend=False,
                legendgroup=legend_group,
            ),
            **rc,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean.tolist(),
            mode="lines",
            name=label,
            line=dict(color=_rgba(color_hex, 1.0), width=3, dash=dash),
            legendgroup=legend_group,
            showlegend=True,
        ),
        **rc,
    )


def _style_prob_axes(fig, n_layers: int, row: int = None, col: int = None):
    """Apply standard axis styling to a probability subplot."""
    rc = dict(row=row, col=col) if row is not None else {}
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_layers)),
        ticktext=[str(i) for i in range(n_layers)],
        range=[-0.5, n_layers - 0.5],
        showgrid=False,
        tickfont=dict(size=11),
        title=dict(text="Layer Index", font=dict(size=14)),
        **rc,
    )
    fig.update_yaxes(
        range=[0, 1],
        showline=True,
        linewidth=1,
        linecolor="black",
        showgrid=False,
        title=dict(text="Answer Probability", font=dict(size=14)),
        tickfont=dict(size=12),
        **rc,
    )


def _add_intervention_span(fig, layers: List[int], n_layers: int, row: int = None, col: int = None):
    """Add a shaded vertical region marking intervention layers."""
    if not layers:
        return
    lo = min(layers) - 0.5
    hi = max(layers) + 0.5
    rc = dict(row=row, col=col) if row is not None else {}
    fig.add_vrect(
        x0=lo,
        x1=hi,
        fillcolor="rgba(180,180,180,0.15)",
        line=dict(width=1, color="rgba(150,150,150,0.4)"),
        **rc,
    )
    mid = (lo + hi) / 2
    # Determine axis references for annotation placement
    if row is not None and row > 1:
        xref, yref = f"x{row}", f"y{row}"
    else:
        xref, yref = "x", "y"
    fig.add_annotation(
        x=mid,
        y=1.02,
        xref=xref,
        yref=yref,
        text="<b>Intervention</b>",
        showarrow=False,
        font=dict(size=11, color="gray"),
    )


def _final_search_evaluation(search_trace: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluations = search_trace.get("evaluations", [])
    if not evaluations:
        raise ValueError("search_trace has no probability evaluations")

    final_id = search_trace.get("final_evaluation_id")
    if isinstance(final_id, int) and 0 <= final_id < len(evaluations):
        return evaluations[final_id]

    selected_layers = search_trace.get("selected_layers")
    if selected_layers is not None:
        for evaluation in reversed(evaluations):
            if evaluation.get("patch_layers") == selected_layers:
                return evaluation
    return evaluations[-1]


def _saved_gap_condition_to_tensors(
    condition: Mapping[str, Any],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Restore ``[layer, source/target]`` statistics from JSON."""
    source = condition["source"]
    target = condition["target"]
    return tuple(
        torch.tensor(list(zip(source[statistic], target[statistic])), dtype=torch.float32)
        for statistic in ("mean", "std", "median")
    )


def _plot_saved_probability_gap(
    search_trace: Mapping[str, Any],
    output_path: Optional[str],
    *,
    show_onset: bool = True,
) -> go.Figure:
    plot_data = search_trace["probability_gap_plot"]
    patching_layers = list(plot_data.get("patch_layers", search_trace.get("selected_layers", [])))
    baseline = _saved_gap_condition_to_tensors(plot_data["baseline"])
    patched = _saved_gap_condition_to_tensors(plot_data["patched"])
    if show_onset:
        try:
            onset = resolve_difference_onset(search_trace)
        except ValueError:
            onset = None
    else:
        onset = None
    onset_annotation = None
    if onset is not None and not onset.get("used_fallback"):
        onset_annotation = (
            f"<b>onset L{onset['layer']}</b><br>effect {onset['effect_at_layer']:.2f} ≥ {onset['threshold']:.2f}"
        )
    return plot_probability_gap(
        baseline,
        patched,
        patching_layers=patching_layers,
        direction=search_trace.get("task_direction", "prior_to_context"),
        site_label="MHA output",
        n_layers=int(search_trace["n_layers"]),
        output_path=output_path,
        onset_layer=onset["layer"] if onset is not None else None,
        onset_is_fallback=(bool(onset.get("used_fallback")) if onset is not None else False),
        onset_annotation=onset_annotation,
    )


def _plot_full_answer_search_trace(
    search_trace: Mapping[str, Any],
    output_path: Optional[str],
) -> go.Figure:
    """Plot only the saved full-answer margin/effect values."""
    evaluations = list(search_trace.get("evaluations", []))
    if not evaluations:
        raise ValueError("search_trace has no full-answer evaluations")
    selected_layers = list(search_trace.get("selected_layers", []))
    threshold = float(search_trace.get("parameters", {}).get("retention_threshold", 0.9))
    evaluation_ids = [evaluation["evaluation_id"] for evaluation in evaluations]
    hover_details = [
        (
            f"phase={evaluation['phase']}<br>"
            f"patch=L{_format_layers(evaluation['patch_layers']) or 'none'}<br>"
            f"cache_hit={evaluation.get('cache_hit', False)}<br>"
            f"samples={evaluation.get('n_samples', 'unknown')}"
        )
        for evaluation in evaluations
    ]

    fig = sp.make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.17,
        subplot_titles=(
            "Complete-answer probability scores",
            f"Baseline-normalized retained effect (selected: L{_format_layers(selected_layers)})",
        ),
    )
    for label, field, color, dash in (
        ("Source answer q", "source_mean_logprob", _SRC_HEX, "solid"),
        ("Target answer q", "target_mean_logprob", _TGT_HEX, "solid"),
        ("Margin m(S)", "margin", "#2A9D8F", "dash"),
    ):
        fig.add_trace(
            go.Scatter(
                x=evaluation_ids,
                y=[evaluation[field] for evaluation in evaluations],
                text=hover_details,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=7),
                hovertemplate=("%{text}<br>Evaluation=%{x}<br>Value=%{y:.5f}<extra>%{fullData.name}</extra>"),
            ),
            row=1,
            col=1,
        )

    normalized_values = [evaluation.get("normalized_effect") for evaluation in evaluations]
    fig.add_trace(
        go.Scatter(
            x=evaluation_ids,
            y=normalized_values,
            text=hover_details,
            mode="lines+markers",
            name="Normalized effect R(S)",
            line=dict(color="#6F4EAD", width=3),
            marker=dict(size=8),
            hovertemplate=("%{text}<br>Evaluation=%{x}<br>R(S)=%{y:.5f}<extra>%{fullData.name}</extra>"),
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=threshold,
        line=dict(color="#6C757D", width=1.5, dash="dash"),
        annotation_text=f"selection threshold R={threshold:g}",
        annotation_position="top right",
        row=2,
        col=1,
    )

    final_id = search_trace.get("final_evaluation_id")
    if isinstance(final_id, int):
        for row in (1, 2):
            fig.add_vline(
                x=final_id,
                line=dict(color="rgba(80,80,80,0.45)", width=1, dash="dot"),
                row=row,
                col=1,
            )

    fig.update_xaxes(title_text="Evaluation order", dtick=1, row=1, col=1)
    fig.update_xaxes(title_text="Evaluation order", dtick=1, row=2, col=1)
    fig.update_yaxes(title_text="Mean log-probability / margin", row=1, col=1)
    fig.update_yaxes(title_text="R(S)", row=2, col=1)
    fig.update_layout(
        title=dict(text="<b>Important-layer search trace</b>"),
        font=dict(family=_FONT_FAMILY, size=13),
        width=1100,
        height=760,
        margin=dict(l=75, r=35, t=80, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".html", ".htm"}:
            fig.write_html(str(path), include_plotlyjs=True, full_html=True)
        else:
            fig.write_image(str(path))
    return fig


def plot_layer_search_trace(
    search_trace: Mapping[str, Any],
    output_path: Optional[str] = None,
    *,
    show_onset: bool = True,
) -> go.Figure:
    """Plot values already collected by ``auto_search`` without new inference.

    The first panel shows the complete per-readout-layer statistics for the
    final selected patch range. The second panel shows every candidate
    evaluation in execution order, including thresholds used by the search.
    """
    if "probability_gap_plot" in search_trace:
        return _plot_saved_probability_gap(
            search_trace,
            output_path,
            show_onset=show_onset,
        )
    if search_trace.get("objective") == "full_answer_logprob_margin":
        return _plot_full_answer_search_trace(search_trace, output_path)

    evaluations = list(search_trace.get("evaluations", []))
    final_evaluation = _final_search_evaluation(search_trace)
    selected_layers = list(search_trace.get("selected_layers", final_evaluation.get("patch_layers", [])))

    source = final_evaluation["source"]
    target = final_evaluation["target"]
    n_layers = int(search_trace.get("n_layers", len(source["mean"])))
    if len(source["mean"]) != n_layers or len(target["mean"]) != n_layers:
        raise ValueError("final search probabilities do not match trace n_layers")

    fig = sp.make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.16,
        subplot_titles=(
            f"Final selected patch range: L{_format_layers(selected_layers)}",
            "Search evaluations (hover for phase and patched layers)",
        ),
    )
    layer_x = list(range(n_layers))

    for label, stats, color in (
        ("Source answer", source, _SRC_HEX),
        ("Target answer", target, _TGT_HEX),
    ):
        customdata = list(zip(stats["std"], stats["median"]))
        fig.add_trace(
            go.Scatter(
                x=layer_x,
                y=stats["mean"],
                customdata=customdata,
                mode="lines+markers",
                name=f"{label} mean",
                line=dict(color=_rgba(color, 1.0), width=3),
                marker=dict(size=5),
                hovertemplate=(
                    "Readout layer %{x}<br>Mean=%{y:.4f}<br>"
                    "Std=%{customdata[0]:.4f}<br>Median=%{customdata[1]:.4f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=layer_x,
                y=stats["median"],
                mode="lines",
                name=f"{label} median",
                line=dict(color=_rgba(color, 0.65), width=2, dash="dot"),
                hovertemplate=("Readout layer %{x}<br>Median=%{y:.4f}<extra>%{fullData.name}</extra>"),
            ),
            row=1,
            col=1,
        )

    for layer in selected_layers:
        fig.add_vrect(
            x0=layer - 0.5,
            x1=layer + 0.5,
            fillcolor="rgba(120,120,120,0.08)",
            line_width=0,
            row=1,
            col=1,
        )

    evaluation_ids = [evaluation["evaluation_id"] for evaluation in evaluations]
    hover_details = [
        (
            f"phase={evaluation['phase']}<br>"
            f"patch=L{_format_layers(evaluation['patch_layers']) or 'none'}<br>"
            f"cache_hit={evaluation.get('cache_hit', False)}"
        )
        for evaluation in evaluations
    ]
    summary_series = (
        ("Max source", "max_source_probability", _SRC_HEX, "solid"),
        ("Output source", "output_source_probability", _SRC_HEX, "dash"),
        ("Output target", "output_target_probability", _TGT_HEX, "dash"),
        ("Output margin", "output_margin", "#2A9D8F", "dot"),
    )
    for label, field, color, dash in summary_series:
        fig.add_trace(
            go.Scatter(
                x=evaluation_ids,
                y=[evaluation[field] for evaluation in evaluations],
                text=hover_details,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=7),
                hovertemplate=("%{text}<br>Evaluation=%{x}<br>Value=%{y:.4f}<extra>%{fullData.name}</extra>"),
            ),
            row=2,
            col=1,
        )

    threshold_specs = (
        ("upper_threshold_value", "Upper threshold", "#6C757D"),
        ("lower_threshold_value", "Lower threshold", "#ADB5BD"),
    )
    for key, label, color in threshold_specs:
        value = search_trace.get(key)
        if value is not None:
            fig.add_hline(
                y=float(value),
                line=dict(color=color, width=1, dash="dash"),
                annotation_text=label,
                annotation_position="top right",
                row=2,
                col=1,
            )

    eps = search_trace.get("parameters", {}).get("eps")
    if eps is not None:
        fig.add_hline(
            y=float(eps),
            line=dict(color="#2A9D8F", width=1, dash="dot"),
            annotation_text="Required output margin (eps)",
            annotation_position="bottom right",
            row=2,
            col=1,
        )

    fig.update_xaxes(
        title_text="Readout layer",
        tickmode="linear",
        dtick=1,
        range=[-0.5, n_layers - 0.5],
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Answer probability", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Evaluation order", dtick=1, row=2, col=1)
    fig.update_yaxes(title_text="Probability / margin", range=[-1, 1], row=2, col=1)
    fig.update_layout(
        title=dict(text="<b>Important-layer search trace</b>"),
        font=dict(family=_FONT_FAMILY, size=13),
        width=1100,
        height=760,
        margin=dict(l=75, r=35, t=80, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".html", ".htm"}:
            fig.write_html(str(path), include_plotlyjs=True, full_html=True)
        else:
            fig.write_image(str(path))
    return fig


# ---------------------------------------------------------------------------
# Plot 1: Before/after overlay
# ---------------------------------------------------------------------------


def plot_overlay(
    probs_baseline: Tuple[Tensor, Tensor, Tensor],
    probs_patched: Tuple[Tensor, Tensor, Tensor],
    patching_layers: List[int],
    direction: str = "prior_to_context",
    site_label: str = "MHA output",
    n_layers: int = 32,
    output_path: Optional[str] = None,
) -> go.Figure:
    """
    Before/after overlay: 4 lines on one plot.

    - Solid lines = with patching (source answer blue, target answer red)
    - Dashed lines = no patching baseline
    - Vertical span marks intervention layers
    - Title states the hypothesis; legend labels explain expected direction
    """
    src_label, tgt_label = _direction_labels(direction)
    title = _hypothesis_title(direction, site_label, patching_layers)
    fig = go.Figure()

    # Baseline (dashed) - source answer
    _add_prob_lines(
        fig,
        (probs_baseline[0][:, 0], probs_baseline[1][:, 0], probs_baseline[2][:, 0]),
        n_layers,
        _SRC_HEX,
        f"{src_label} - no patch",
        dash="dash",
        show_band=False,
        legend_group="src",
    )
    # Baseline (dashed) - target answer
    _add_prob_lines(
        fig,
        (probs_baseline[0][:, 1], probs_baseline[1][:, 1], probs_baseline[2][:, 1]),
        n_layers,
        _TGT_HEX,
        f"{tgt_label} - no patch",
        dash="dash",
        show_band=False,
        legend_group="tgt",
    )
    # Patched (solid) - source answer (should ↑)
    _add_prob_lines(
        fig,
        (probs_patched[0][:, 0], probs_patched[1][:, 0], probs_patched[2][:, 0]),
        n_layers,
        _SRC_HEX,
        f"{src_label} - patched (should ↑)",
        dash="solid",
        show_band=True,
        legend_group="src",
    )
    # Patched (solid) - target answer (should ↓)
    _add_prob_lines(
        fig,
        (probs_patched[0][:, 1], probs_patched[1][:, 1], probs_patched[2][:, 1]),
        n_layers,
        _TGT_HEX,
        f"{tgt_label} - patched (should ↓)",
        dash="solid",
        show_band=True,
        legend_group="tgt",
    )

    _add_intervention_span(fig, patching_layers, n_layers)
    _style_prob_axes(fig, n_layers)

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16)),
        font=dict(family=_FONT_FAMILY, size=14),
        width=1100,
        height=400,
        margin=dict(l=70, r=30, t=70, b=55),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bordercolor="black",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=13),
        ),
    )

    if output_path:
        import os

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if Path(output_path).suffix.lower() in {".html", ".htm"}:
            fig.write_html(output_path, include_plotlyjs=True, full_html=True)
        else:
            fig.write_image(output_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Delta bar chart
# ---------------------------------------------------------------------------


def plot_delta_bar(
    probs_baseline: Tuple[Tensor, Tensor, Tensor],
    probs_patched: Tuple[Tensor, Tensor, Tensor],
    patching_layers: List[int],
    direction: str = "prior_to_context",
    site_label: str = "MHA output",
    n_layers: int = 32,
    output_path: Optional[str] = None,
) -> go.Figure:
    """
    Bar chart of ΔP = P(answer|patched) — P(answer|baseline).

    Aggregates over the patching layers (mean of medians in the intervention
    region). Two bars: source answer (positive = success) and target answer
    (negative = success).
    """
    src_label, tgt_label = _direction_labels(direction)
    title = _hypothesis_title(direction, site_label, patching_layers)

    # Use medians for aggregation
    baseline_med = probs_baseline[2]  # [L, 2]
    patched_med = probs_patched[2]

    if patching_layers:
        idx = torch.tensor(patching_layers)
        bl_src = baseline_med[idx, 0].mean().item()
        bl_tgt = baseline_med[idx, 1].mean().item()
        pa_src = patched_med[idx, 0].mean().item()
        pa_tgt = patched_med[idx, 1].mean().item()
    else:
        bl_src = baseline_med[:, 0].mean().item()
        bl_tgt = baseline_med[:, 1].mean().item()
        pa_src = patched_med[:, 0].mean().item()
        pa_tgt = patched_med[:, 1].mean().item()

    delta_src = pa_src - bl_src
    delta_tgt = pa_tgt - bl_tgt

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[f"{src_label}<br>(should ↑)"],
            y=[delta_src],
            marker_color=_rgba(_SRC_HEX, 0.85),
            text=[f"{delta_src:+.3f}"],
            textposition="outside",
            textfont=dict(size=16, color=_rgba(_SRC_HEX, 1.0)),
            name=src_label,
            width=0.4,
        )
    )
    fig.add_trace(
        go.Bar(
            x=[f"{tgt_label}<br>(should ↓)"],
            y=[delta_tgt],
            marker_color=_rgba(_TGT_HEX, 0.85),
            text=[f"{delta_tgt:+.3f}"],
            textposition="outside",
            textfont=dict(size=16, color=_rgba(_TGT_HEX, 1.0)),
            name=tgt_label,
            width=0.4,
        )
    )

    # Zero line
    fig.add_hline(y=0, line=dict(color="black", width=1))

    y_max = max(abs(delta_src), abs(delta_tgt)) * 1.5
    y_max = max(y_max, 0.1)  # floor so the plot isn't empty

    fig.update_layout(
        title=dict(text=f"<b>{title}: ΔP</b>", font=dict(size=16)),
        font=dict(family=_FONT_FAMILY, size=14),
        yaxis=dict(
            title="ΔP  (patched − baseline)",
            range=[-y_max, y_max],
            zeroline=True,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
        ),
        width=450,
        height=400,
        margin=dict(l=70, r=30, t=70, b=55),
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.35,
    )

    if output_path:
        import os

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.write_image(output_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Probability gap
# ---------------------------------------------------------------------------


def plot_probability_gap(
    probs_baseline: Tuple[Tensor, Tensor, Tensor],
    probs_patched: Tuple[Tensor, Tensor, Tensor],
    patching_layers: List[int],
    direction: str = "prior_to_context",
    site_label: str = "MHA output",
    n_layers: int = 32,
    output_path: Optional[str] = None,
    onset_layer: Optional[int] = None,
    onset_is_fallback: bool = False,
    onset_annotation: Optional[str] = None,
) -> go.Figure:
    """
    Plot P(source answer) — P(target answer) for baseline vs patched.

    A single line per condition. When baseline is negative and patched is
    positive, patching successfully flipped the model's answer. The shaded
    area between the two curves shows the effect magnitude.
    """
    src_label, tgt_label = _direction_labels(direction)

    # Compute gap = P(source) - P(target) using means
    gap_bl = (probs_baseline[0][:, 0] - probs_baseline[0][:, 1]).tolist()
    gap_pa = (probs_patched[0][:, 0] - probs_patched[0][:, 1]).tolist()

    # Also compute gap from medians for reference line
    gap_bl_med = (probs_baseline[2][:, 0] - probs_baseline[2][:, 1]).tolist()
    gap_pa_med = (probs_patched[2][:, 0] - probs_patched[2][:, 1]).tolist()

    x = list(range(n_layers))

    # Blue for patched, red for no-patching baseline
    _PATCHED_HEX = _SRC_HEX
    _BASELINE_HEX = _TGT_HEX

    fig = go.Figure()

    # Background tint: blue above zero (succeeded), red below zero (failed)
    fig.add_hrect(
        y0=0,
        y1=1,
        fillcolor=_rgba(_PATCHED_HEX, 0.04),
        line=dict(width=0),
    )
    fig.add_hrect(
        y0=-1,
        y1=0,
        fillcolor=_rgba(_BASELINE_HEX, 0.04),
        line=dict(width=0),
    )

    # Shaded area between the two curves
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=gap_pa + gap_bl[::-1],
            fill="toself",
            fillcolor=_rgba(_PATCHED_HEX, 0.10),
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Baseline gap line (red)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=gap_bl,
            mode="lines",
            line=dict(color=_rgba(_BASELINE_HEX, 1.0), width=3),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=gap_bl_med,
            mode="lines",
            line=dict(color=_rgba(_BASELINE_HEX, 0.6), width=2, dash="dot"),
            showlegend=False,
        )
    )

    # Patched gap line (blue)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=gap_pa,
            mode="lines",
            line=dict(color=_rgba(_PATCHED_HEX, 1.0), width=3),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=gap_pa_med,
            mode="lines",
            line=dict(color=_rgba(_PATCHED_HEX, 0.6), width=2, dash="dot"),
            showlegend=False,
        )
    )

    # Legend entries: source/target color keys + Mean/Median line style keys
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=src_label,
            marker=dict(color=_rgba(_PATCHED_HEX, 1.0), size=10, symbol="circle"),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=tgt_label,
            marker=dict(color=_rgba(_BASELINE_HEX, 1.0), size=10, symbol="circle"),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Mean",
            line=dict(color="black", width=3),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Median",
            line=dict(color="black", width=2, dash="dot"),
            showlegend=True,
        )
    )

    # Note: The remaining part of this function (axes styling, layout updates, and onset annotation)
    # is not visible in the provided screenshots. The code above captures all visible portions.
    # ... 接 plot_probability_gap 函数中图例部分之后 ...

    # Zero line - the decision boundary
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))

    if onset_layer is not None:
        if not 0 <= onset_layer < n_layers:
            raise ValueError(f"onset_layer must be within [0, {n_layers - 1}], got {onset_layer}")
        onset_color = "#6C757D" if onset_is_fallback else "#2A9D8F"
        onset_label = (
            f"<b>target L{onset_layer} (range end)</b>"
            if onset_is_fallback
            else onset_annotation or f"<b>onset L{onset_layer}</b>"
        )
        fig.add_vline(
            x=onset_layer,
            line=dict(color=onset_color, width=2, dash="dash"),
            annotation_text=onset_label,
            annotation_position="top left",
            annotation_font=dict(color=onset_color, size=12),
        )

    # Region labels: centered, bold, colored to match their line
    mid_x = n_layers / 2
    fig.add_annotation(
        x=mid_x,
        y=0.15,
        xref="x",
        yref="y",
        text="<b>Patched</b>",
        showarrow=False,
        font=dict(size=32, color=_rgba(_PATCHED_HEX, 0.25), family=_FONT_FAMILY),
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=mid_x,
        y=-0.15,
        xref="x",
        yref="y",
        text="<b>No Patching</b>",
        showarrow=False,
        font=dict(size=32, color=_rgba(_BASELINE_HEX, 0.25), family=_FONT_FAMILY),
        xanchor="center",
        yanchor="middle",
    )

    # Highlight patching layers: bold colored text + background rectangles
    _patching_set = set(patching_layers)
    tick_text = []
    for i in range(n_layers):
        if i in _patching_set:
            tick_text.append(f"<b>{i}</b>")
        else:
            tick_text.append(str(i))

    # Axes
    fig.update_xaxes(
        tickmode="array",
        tickvals=x,
        ticktext=tick_text,
        range=[-0.5, n_layers - 0.5],
        showgrid=False,
        tickfont=dict(size=14),
        title=dict(text="<b>Layer Index</b>", font=dict(size=18)),
    )

    # Add background highlight rectangles behind patching-layer tick labels
    if patching_layers:
        x_domain = fig.layout.xaxis.domain or (0, 1)
        plot_left = x_domain[0] if x_domain else 0
        plot_right = x_domain[1] if x_domain else 1
        plot_width = plot_right - plot_left
        x_range_lo = -0.5
        x_range_hi = n_layers - 0.5
        x_span = x_range_hi - x_range_lo

        for layer_idx in patching_layers:
            frac_left = plot_left + plot_width * (layer_idx - 0.5 - x_range_lo) / x_span
            frac_right = plot_left + plot_width * (layer_idx + 0.5 - x_range_lo) / x_span
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=frac_left,
                x1=frac_right,
                y0=-0.001,
                y1=-0.075,
                fillcolor=_rgba(_PATCHED_HEX, 0.15),
                line=dict(width=0),
                layer="below",
            )

    fig.update_yaxes(
        range=[-1, 1],
        showline=True,
        linewidth=1,
        linecolor="black",
        showgrid=False,
        zeroline=False,
        title=dict(
            text="<b>Δ(Answer Probability)</b>",
            font=dict(size=20),
        ),
        tickfont=dict(size=22),
    )

    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=14),
        width=1100,
        height=340,
        margin=dict(l=80, r=30, t=30, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bordercolor="black",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=16, family=f"{_FONT_FAMILY}"),
        ),
    )

    # Bold legend item names
    for trace in fig.data:
        if trace.name:
            trace.name = f"<b>{trace.name}</b>"

    if output_path:
        import os

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if Path(output_path).suffix.lower() in {".html", ".htm"}:
            fig.write_html(output_path, include_plotlyjs=True, full_html=True)
        else:
            fig.write_image(output_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 4: 2x2 grid comparison
# ---------------------------------------------------------------------------


def plot_grid_comparison(
    results: List[dict],
    n_layers: int = 32,
    output_path: Optional[str] = None,
) -> go.Figure:
    """
    2x2 (or Nx2) grid: each row is a direction, columns are baseline vs patched.

    Parameters
    ----------
    results : list of dict
        Each dict has:
        - ``direction`` : str, e.g. "prior_to_context"
        - ``probs_baseline`` : (mean, std, median) tuple
        - ``probs_patched`` : (mean, std, median) tuple
        - ``patching_layers`` : list of int
        - ``site_label`` : str, e.g. "MHA output"
    n_layers : int
    output_path : str or None
    """
    n_rows = len(results)
    titles = []
    for res in results:
        src_label, tgt_label = _direction_labels(res["direction"])
        layer_str = _format_layers(res["patching_layers"])
        titles.append(f"<b>No patching</b> ({src_label} vs {tgt_label})")
        titles.append(f"<b>Patched {res['site_label']} at L{layer_str}</b>")

    fig = sp.make_subplots(
        rows=n_rows,
        cols=2,
        shared_yaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.18,
        horizontal_spacing=0.06,
    )

    for row_idx, res in enumerate(results, start=1):
        src_label, tgt_label = _direction_labels(res["direction"])
        probs_bl = res["probs_baseline"]
        probs_pa = res["probs_patched"]
        layers = res["patching_layers"]

        # Column 1: baseline
        _add_prob_lines(
            fig,
            (probs_bl[0][:, 0], probs_bl[1][:, 0], probs_bl[2][:, 0]),
            n_layers,
            _SRC_HEX,
            src_label,
            row=row_idx,
            col=1,
            legend_group=f"src_{row_idx}",
            show_band=True,
        )
        _add_prob_lines(
            fig,
            (probs_bl[0][:, 1], probs_bl[1][:, 1], probs_bl[2][:, 1]),
            n_layers,
            _TGT_HEX,
            tgt_label,
            row=row_idx,
            col=1,
            legend_group=f"tgt_{row_idx}",
            show_band=True,
        )
        _style_prob_axes(fig, n_layers, row=row_idx, col=1)

        # Column 2: patched
        _add_prob_lines(
            fig,
            (probs_pa[0][:, 0], probs_pa[1][:, 0], probs_pa[2][:, 0]),
            n_layers,
            _SRC_HEX,
            src_label,
            row=row_idx,
            col=2,
            legend_group=f"src_{row_idx}",
            show_band=True,
        )
        _add_prob_lines(
            fig,
            (probs_pa[0][:, 1], probs_pa[1][:, 1], probs_pa[2][:, 1]),
            n_layers,
            _TGT_HEX,
            tgt_label,
            row=row_idx,
            col=2,
            legend_group=f"tgt_{row_idx}",
            show_band=True,
        )
        _add_intervention_span(fig, layers, n_layers, row=row_idx, col=2)
        _style_prob_axes(fig, n_layers, row=row_idx, col=2)

        # Only show y-axis title on the left column
        fig.update_yaxes(title=dict(text=""), row=row_idx, col=2)

    # Deduplicate legend: only show first occurrence of each group
    seen = set()
    for trace in fig.data:
        lg = trace.legendgroup or ""
        if lg in seen:
            trace.showlegend = False
        else:
            seen.add(lg)

    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=13),
        width=1100,
        height=350 * n_rows,
        margin=dict(l=70, r=30, t=60, b=55),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            x=0.5,
            y=-0.05,
            xanchor="center",
            yanchor="top",
            orientation="h",
            bordercolor="black",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=14),
        ),
    )
    fig.update_annotations(font=dict(size=15))

    if output_path:
        import os

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.write_image(output_path)
    return fig
