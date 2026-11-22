"""Comparison charts: cross-run, parallel coordinates, location quality."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._helpers import _empty_figure
from .style import CATEGORICAL_COLORS, get_bloom_color

if TYPE_CHECKING:
    from arandu.report.dataset import QAPairRow


def create_cross_run_comparison(qa_pairs: list[QAPairRow], run_a: str, run_b: str) -> go.Figure:
    """Create a 2x2 subplot comparing validation criteria between two runs.

    Each subplot shows overlapping violin plots for one criterion.

    Args:
        qa_pairs: List of QAPairRow objects.
        run_a: First pipeline ID to compare.
        run_b: Second pipeline ID to compare.

    Returns:
        Plotly Figure object.
    """
    criteria = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"]
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=[c.replace("_", " ").title() for c in criteria]
    )

    pairs_a = [qa for qa in qa_pairs if qa.pipeline_id == run_a]
    pairs_b = [qa for qa in qa_pairs if qa.pipeline_id == run_b]

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (row, col), criterion in zip(positions, criteria, strict=False):
        scores_a = [getattr(qa, criterion) for qa in pairs_a if getattr(qa, criterion) is not None]
        scores_b = [getattr(qa, criterion) for qa in pairs_b if getattr(qa, criterion) is not None]

        if scores_a:
            fig.add_trace(
                go.Violin(
                    y=scores_a,
                    name=run_a,
                    marker_color=CATEGORICAL_COLORS[0],
                    legendgroup=run_a,
                    showlegend=(row == 1 and col == 1),
                    box_visible=True,
                    meanline_visible=True,
                ),
                row=row,
                col=col,
            )

        if scores_b:
            fig.add_trace(
                go.Violin(
                    y=scores_b,
                    name=run_b,
                    marker_color=CATEGORICAL_COLORS[1],
                    legendgroup=run_b,
                    showlegend=(row == 1 and col == 1),
                    box_visible=True,
                    meanline_visible=True,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=f"Cross-Run Comparison: {run_a} vs {run_b}",
        template="plotly_white",
        height=600,
        violinmode="overlay",
    )

    return fig


def create_parallel_coordinates_chart(qa_pairs: list[QAPairRow]) -> go.Figure:
    """Create a parallel coordinates chart for QA pair validation dimensions.

    Dimensions: confidence, faithfulness, bloom_calibration, informativeness,
    self_containedness. Color by bloom level.

    Args:
        qa_pairs: List of QAPairRow objects to visualize.

    Returns:
        Plotly Figure object.
    """
    fields = [
        "confidence",
        "faithfulness",
        "bloom_calibration",
        "informativeness",
        "self_containedness",
    ]
    bloom_order = ["remember", "understand", "analyze", "evaluate"]

    # Filter to rows with all values present
    rows = []
    for qa in qa_pairs:
        vals = {f: getattr(qa, f) for f in fields}
        if all(v is not None for v in vals.values()) and qa.bloom_level in bloom_order:
            vals["bloom_idx"] = bloom_order.index(qa.bloom_level)
            rows.append(vals)

    if not rows:
        return _empty_figure("Parallel Coordinates: Validation Dimensions")

    dimensions = [
        {"label": f.replace("_", " ").title(), "values": [r[f] for r in rows], "range": [0, 1]}
        for f in fields
    ]

    fig = go.Figure(
        data=go.Parcoords(
            line={
                "color": [r["bloom_idx"] for r in rows],
                "colorscale": [[i / 3, get_bloom_color(b)] for i, b in enumerate(bloom_order)],
                "showscale": True,
                "cmin": 0,
                "cmax": 3,
                "colorbar": {
                    "title": "Bloom Level",
                    "tickvals": [0, 1, 2, 3],
                    "ticktext": [b.capitalize() for b in bloom_order],
                },
            },
            dimensions=dimensions,
        )
    )

    fig.update_layout(
        title="Multi-dimensional QA Quality Profile (colored by Bloom level)",
        template="plotly_white",
        height=500,
    )

    return fig


def create_location_quality_chart(qa_pairs: list[QAPairRow]) -> go.Figure:
    """Create violin plots of validation scores grouped by location.

    Args:
        qa_pairs: List of QAPairRow objects to visualize.

    Returns:
        Plotly Figure object.
    """
    # Use overall_score grouped by location
    by_location: dict[str, list[float]] = defaultdict(list)
    for qa in qa_pairs:
        if qa.overall_score is not None:
            loc = qa.location or "Unknown"
            by_location[loc].append(qa.overall_score)

    fig = go.Figure()

    for i, (location, scores) in enumerate(sorted(by_location.items())):
        fig.add_trace(
            go.Violin(
                y=scores,
                name=location,
                marker_color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)],
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig.update_layout(
        title="Overall Validation Score Distribution by Recording Location",
        yaxis_title="Overall Score (0-1)",
        template="plotly_white",
        height=450,
    )

    return fig
