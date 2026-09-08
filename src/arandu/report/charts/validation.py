"""Validation charts: score distributions, Bloom heatmap, correlation matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from .style import get_criterion_color

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from arandu.report.dataset import QAPairRow

_CEP_CRITERIA: tuple[str, ...] = (
    "faithfulness",
    "bloom_calibration",
    "informativeness",
    "self_containedness",
)
"""The four CEP QA criteria, in the order they are plotted."""

_VIOLIN_HALF_WIDTH: float = 0.4
"""Half-width, in category units, of the slot a single violin occupies."""


def create_validation_scores_chart(
    qa_pairs: list[QAPairRow],
    criterion_thresholds: Mapping[str, float] | None = None,
) -> go.Figure:
    """Create violin plots showing validation score distributions.

    Uses violin plots instead of box plots to reveal bimodal distributions
    that are common in validation scores.

    Args:
        qa_pairs: List of QAPairRow objects to visualize.
        criterion_thresholds: Optional per-criterion judge gate. Each plotted
            criterion gets a cut line at its own gate; criteria missing from
            the mapping are left unmarked. There is no aggregate cut, so no
            single line spans the whole axis.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    plotted: list[str] = []

    for criterion in _CEP_CRITERIA:
        scores = [getattr(qa, criterion) for qa in qa_pairs if getattr(qa, criterion) is not None]
        if scores:
            plotted.append(criterion)
            fig.add_trace(
                go.Violin(
                    y=scores,
                    name=criterion.replace("_", " ").title(),
                    marker_color=get_criterion_color(criterion),
                    box_visible=True,
                    meanline_visible=True,
                )
            )

    fig.update_layout(
        title="LLM-as-a-Judge Validation Score Distributions",
        yaxis_title="Score (0-1)",
        template="plotly_white",
        height=450,
    )

    if criterion_thresholds:
        _add_criterion_cut_lines(fig, plotted, criterion_thresholds)

    return fig


def create_bloom_validation_heatmap(
    qa_pairs: list[QAPairRow],
    criterion_thresholds: Mapping[str, float] | None = None,
) -> go.Figure:
    """Create a heatmap of mean validation scores by Bloom level and criterion.

    Rows: Bloom levels, Cols: validation criteria. Cells annotated with mean +/- std.
    When a criterion's gate is known, cells whose mean falls below that column's
    own gate are marked with a warning.

    Args:
        qa_pairs: List of QAPairRow objects to visualize.
        criterion_thresholds: Optional per-criterion judge gate. Each column is
            compared against its own gate; columns missing from the mapping are
            never flagged.

    Returns:
        Plotly Figure object.
    """
    bloom_levels = ["remember", "understand", "analyze", "evaluate"]
    criteria = list(_CEP_CRITERIA)
    criteria_labels = [c.replace("_", " ").title() for c in criteria]
    gates = criterion_thresholds or {}

    # Collect scores by bloom level and criterion
    data: dict[str, dict[str, list[float]]] = {b: {c: [] for c in criteria} for b in bloom_levels}

    for qa in qa_pairs:
        if qa.bloom_level in bloom_levels:
            for criterion in criteria:
                score = getattr(qa, criterion)
                if score is not None:
                    data[qa.bloom_level][criterion].append(score)

    # Compute means and build annotation text
    z = []
    text = []
    for level in bloom_levels:
        row_z = []
        row_text = []
        for criterion in criteria:
            scores = data[level][criterion]
            if scores:
                mean = sum(scores) / len(scores)
                std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
                row_z.append(mean)
                cell_text = f"{mean:.2f}\n+/-{std:.2f}"
                gate = gates.get(criterion)
                if gate is not None and mean < gate:
                    cell_text = f"\u26a0 {cell_text}"
                row_text.append(cell_text)
            else:
                row_z.append(0.0)
                row_text.append("N/A")
        z.append(row_z)
        text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=criteria_labels,
            y=[b.capitalize() for b in bloom_levels],
            text=text,
            texttemplate="%{text}",
            colorscale="YlOrRd",
            zmin=0,
            zmax=1,
            colorbar={"title": "Mean Score"},
        )
    )

    fig.update_layout(
        title="Mean Validation Score by Bloom Level and Criterion",
        template="plotly_white",
        height=400,
    )

    return fig


def create_correlation_heatmap(qa_pairs: list[QAPairRow]) -> go.Figure:
    """Create a Pearson correlation matrix across validation criteria and confidence.

    Annotated with r-values. Diverging blue-white-red color scale.

    Args:
        qa_pairs: List of QAPairRow objects to visualize.

    Returns:
        Plotly Figure object.
    """
    fields = [*_CEP_CRITERIA, "confidence"]
    labels = [f.replace("_", " ").title() for f in fields]

    # Collect rows where all values are present
    data: list[list[float]] = []
    for qa in qa_pairs:
        vals = [getattr(qa, f) for f in fields]
        if all(v is not None for v in vals):
            data.append(vals)

    n = len(fields)
    corr = [[0.0] * n for _ in range(n)]

    if len(data) >= 3:
        # Compute Pearson correlation
        columns = list(zip(*data, strict=True))
        for i in range(n):
            for j in range(n):
                corr[i][j] = _pearson_r(columns[i], columns[j])
    else:
        for i in range(n):
            corr[i][i] = 1.0

    # Build annotation text
    text = [[f"{corr[i][j]:.2f}" for j in range(n)] for i in range(n)]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar={"title": "r"},
        )
    )

    fig.update_layout(
        title="Pearson Correlation: Validation Criteria & Confidence",
        template="plotly_white",
        height=500,
        width=600,
    )

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_threshold_line(
    fig: go.Figure,
    threshold: float,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add a dashed red threshold line to a figure.

    Args:
        fig: Plotly Figure to modify in place.
        threshold: Threshold value for the overlay line.
        row: Optional subplot row (1-indexed) for figures with subplots.
        col: Optional subplot column (1-indexed) for figures with subplots.
    """
    kwargs: dict[str, Any] = {
        "y": threshold,
        "line_dash": "dash",
        "line_color": "red",
        "line_width": 2,
        "annotation_text": f"Threshold: {threshold}",
        "annotation_position": "top right",
        "annotation_font_color": "red",
        "annotation_font_size": 11,
    }
    if row is not None:
        kwargs["row"] = row
    if col is not None:
        kwargs["col"] = col
    fig.add_hline(**kwargs)


def _add_criterion_cut_lines(
    fig: go.Figure,
    criteria: list[str],
    thresholds: Mapping[str, float],
) -> None:
    """Overlay one dashed cut line per violin, each at that criterion's own gate.

    The judge verdict is a conjunction of independent per-criterion gates, so a
    single figure-wide ``add_hline`` would draw a cut that does not exist. Each
    segment is therefore confined to the slot of the violin it belongs to.

    Args:
        fig: Plotly Figure to modify in place.
        criteria: Criterion names in the order their violins were added.
        thresholds: Per-criterion gate; criteria absent from it are skipped.
    """
    for slot, criterion in enumerate(criteria):
        threshold = thresholds.get(criterion)
        if threshold is None:
            continue
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=slot - _VIOLIN_HALF_WIDTH,
            x1=slot + _VIOLIN_HALF_WIDTH,
            y0=threshold,
            y1=threshold,
            line={"color": "red", "dash": "dash", "width": 2},
        )
        fig.add_annotation(
            x=slot,
            y=threshold,
            text=f"Gate: {threshold:g}",
            showarrow=False,
            yshift=10,
            font={"color": "red", "size": 10},
        )


def _pearson_r(x: tuple[float, ...], y: tuple[float, ...]) -> float:
    """Compute Pearson correlation coefficient between two sequences.

    Args:
        x: First sequence of values.
        y: Second sequence of values.

    Returns:
        Pearson r value, or 0.0 if computation fails.
    """
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
    den_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    den_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)
