"""Multi-hop reasoning chart."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from .style import CATEGORICAL_COLORS

if TYPE_CHECKING:
    from arandu.report.dataset import QAPairRow


def create_multihop_chart(qa_pairs: list[QAPairRow]) -> go.Figure:
    """Create a horizontal bar chart showing multi-hop vs single-hop question counts.

    Uses horizontal bars instead of pie chart for more accurate visual comparison
    (Cleveland & McGill 1984).

    Args:
        qa_pairs: List of QAPairRow objects to visualize.

    Returns:
        Plotly Figure object.
    """
    multihop = sum(1 for qa in qa_pairs if qa.is_multi_hop)
    single = sum(1 for qa in qa_pairs if not qa.is_multi_hop)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Multi-hop", "Single-hop"],
            x=[multihop, single],
            orientation="h",
            marker_color=[CATEGORICAL_COLORS[0], CATEGORICAL_COLORS[1]],
            text=[multihop, single],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Reasoning Complexity: Multi-hop vs Single-hop",
        xaxis_title="Number of QA Pairs",
        template="plotly_white",
        height=300,
    )

    return fig
