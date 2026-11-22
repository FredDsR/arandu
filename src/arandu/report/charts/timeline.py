"""Timeline and pipeline flow charts: overview, run timeline, funnel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._helpers import _empty_figure
from .style import CATEGORICAL_COLORS

if TYPE_CHECKING:
    from arandu.report.dataset import RunSummaryRow
    from arandu.report.schemas import FunnelData


def create_pipeline_overview_chart(runs: list[RunSummaryRow]) -> go.Figure:
    """Create a bar chart showing success rates and durations across pipeline runs.

    Args:
        runs: List of RunSummaryRow objects to visualize.

    Returns:
        Plotly Figure object.
    """
    pipeline_ids = []
    success_rates = []
    durations = []

    for run in runs:
        if run.success_rate is not None:
            pipeline_ids.append(run.pipeline_id)
            success_rates.append(run.success_rate)
            durations.append(run.duration_seconds or 0)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Success Rate by Run", "Processing Duration (seconds)"),
    )

    fig.add_trace(
        go.Bar(
            x=pipeline_ids,
            y=success_rates,
            name="Success Rate (%)",
            marker_color=CATEGORICAL_COLORS[0],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=pipeline_ids,
            y=durations,
            name="Duration (s)",
            marker_color=CATEGORICAL_COLORS[1],
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Pipeline ID", row=1, col=1)
    fig.update_xaxes(title_text="Pipeline ID", row=1, col=2)
    fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Duration (seconds)", row=1, col=2)

    fig.update_layout(
        height=400,
        showlegend=False,
        template="plotly_white",
        title_text="Pipeline Run Success Rate & Processing Duration",
    )

    return fig


def create_run_timeline_chart(runs: list[RunSummaryRow]) -> go.Figure:
    """Create a timeline chart showing run progression over time.

    X: created_at, Y: success_rate. Markers sized by total_items.

    Args:
        runs: List of RunSummaryRow objects to visualize.

    Returns:
        Plotly Figure object.
    """
    dated_runs = [r for r in runs if r.created_at and r.success_rate is not None]
    dated_runs.sort(key=lambda r: r.created_at or "")

    fig = go.Figure()

    if dated_runs:
        fig.add_trace(
            go.Scatter(
                x=[r.created_at for r in dated_runs],
                y=[r.success_rate for r in dated_runs],
                mode="lines+markers",
                marker={
                    "size": [max(8, min(30, (r.total_items or 1) * 2)) for r in dated_runs],
                    "color": CATEGORICAL_COLORS[0],
                },
                text=[r.pipeline_id for r in dated_runs],
                hovertemplate=(
                    "<b>%{text}</b><br>Success Rate: %{y:.1f}%<br>Date: %{x}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Pipeline Run Timeline (marker size = item count)",
        xaxis_title="Date",
        yaxis_title="Success Rate (%)",
        yaxis_range=[0, 100],
        template="plotly_white",
        height=400,
    )

    return fig


def create_funnel_chart(funnel_data: FunnelData) -> go.Figure:
    """Create a Sankey diagram showing data flow through pipeline stages.

    Visualizes the number of items at each stage and drop-off rates
    between stages. Helps identify where data is lost in the pipeline.

    Args:
        funnel_data: FunnelData containing ordered stages with counts.

    Returns:
        Plotly Figure with Sankey diagram.
    """
    stages = funnel_data.stages
    if not stages:
        return _empty_figure("Data Processing Funnel")

    n = len(stages)
    drop_node_idx = n  # single shared sink for all drops

    node_labels = [f"{s.label} ({s.count})" for s in stages] + ["Failed/Invalid"]
    node_colors = ["#029E73"] * n + ["#CC3311"]

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []

    for i in range(n - 1):
        next_stage = stages[i + 1]
        # Forward (success) flow
        if next_stage.count > 0:
            sources.append(i)
            targets.append(i + 1)
            values.append(next_stage.count)
            link_colors.append("rgba(2, 158, 115, 0.4)")
        # Drop flow (items lost between stage i and stage i+1)
        if next_stage.drop_count > 0:
            sources.append(i)
            targets.append(drop_node_idx)
            values.append(next_stage.drop_count)
            link_colors.append("rgba(204, 51, 17, 0.4)")

    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "label": node_labels,
                    "color": node_colors,
                    "pad": 15,
                    "thickness": 20,
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": link_colors,
                },
            )
        ]
    )

    fig.update_layout(
        title="Data Processing Funnel",
        height=400,
        template="plotly_white",
    )

    return fig
