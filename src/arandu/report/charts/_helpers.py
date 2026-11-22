"""Shared helper functions for chart modules."""

from __future__ import annotations

import plotly.graph_objects as go


def _empty_figure(title: str) -> go.Figure:
    """Create an empty figure with a title for when there is no data.

    Args:
        title: Figure title.

    Returns:
        Empty Plotly Figure.
    """
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        annotations=[
            {
                "text": "No data available",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16, "color": "#949494"},
            }
        ],
    )
    return fig
