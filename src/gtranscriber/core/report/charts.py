"""Interactive Plotly chart builders for HTML reports and PNG export.

All chart functions accept flat row lists from ReportDataset instead of
nested RunReport objects, enabling both server-side rendering (PNG export)
and client-side JS mirroring.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .style import CATEGORICAL_COLORS, get_bloom_color, get_criterion_color

if TYPE_CHECKING:
    from .api_schemas import FunnelData
    from .dataset import QAPairRow, RunSummaryRow, TranscriptionRow


# ---------------------------------------------------------------------------
# Refactored existing charts
# ---------------------------------------------------------------------------


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


def create_bloom_distribution_chart(qa_pairs: list[QAPairRow]) -> go.Figure:
    """Create a stacked bar chart showing Bloom's taxonomy distribution by run.

    Args:
        qa_pairs: List of QAPairRow objects to visualize.

    Returns:
        Plotly Figure object.
    """
    bloom_levels = ["remember", "understand", "analyze", "evaluate"]
    by_run: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(bloom_levels, 0))

    for qa in qa_pairs:
        if qa.bloom_level and qa.bloom_level in bloom_levels:
            by_run[qa.pipeline_id][qa.bloom_level] += 1

    pipeline_ids = sorted(by_run.keys())
    fig = go.Figure()

    for level in bloom_levels:
        fig.add_trace(
            go.Bar(
                name=level.capitalize(),
                x=pipeline_ids,
                y=[by_run[pid][level] for pid in pipeline_ids],
                marker_color=get_bloom_color(level),
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Bloom's Taxonomy Level Distribution per Pipeline Run",
        xaxis_title="Pipeline ID",
        yaxis_title="Number of QA Pairs",
        template="plotly_white",
        height=450,
    )

    return fig


def create_validation_scores_chart(
    qa_pairs: list[QAPairRow],
    threshold: float | None = None,
) -> go.Figure:
    """Create violin plots showing validation score distributions.

    Uses violin plots instead of box plots to reveal bimodal distributions
    that are common in validation scores.

    Args:
        qa_pairs: List of QAPairRow objects to visualize.
        threshold: Optional validation threshold for overlay line.

    Returns:
        Plotly Figure object.
    """
    criteria = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"]
    fig = go.Figure()

    for criterion in criteria:
        scores = [getattr(qa, criterion) for qa in qa_pairs if getattr(qa, criterion) is not None]
        if scores:
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

    if threshold is not None:
        _add_threshold_line(fig, threshold)

    return fig


def create_confidence_distribution_chart(qa_pairs: list[QAPairRow]) -> go.Figure:
    """Create a histogram showing confidence score distribution.

    Args:
        qa_pairs: List of QAPairRow objects to visualize.

    Returns:
        Plotly Figure object.
    """
    scores = [qa.confidence for qa in qa_pairs if qa.confidence is not None]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=30,
            marker_color=CATEGORICAL_COLORS[0],
            name="Confidence Score",
        )
    )

    fig.update_layout(
        title="Generation Confidence Score Distribution",
        xaxis_title="Confidence (0-1)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
    )

    return fig


def create_transcription_quality_chart(
    transcriptions: list[TranscriptionRow],
    threshold: float | None = None,
) -> go.Figure:
    """Create histograms showing transcription quality score distributions.

    Args:
        transcriptions: List of TranscriptionRow objects to visualize.
        threshold: Optional quality threshold for overlay line on overall score subplot.

    Returns:
        Plotly Figure object.
    """
    score_fields = [
        ("overall_quality", "Overall Score"),
        ("script_match", "Script Match"),
        ("repetition", "Repetition"),
        ("segment_quality", "Segment Quality"),
        ("content_density", "Content Density"),
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[label for _, label in score_fields] + ["Validity Rate"],
    )

    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

    for (row, col), (field, _label) in zip(positions, score_fields, strict=False):
        scores = [getattr(t, field) for t in transcriptions if getattr(t, field) is not None]
        if scores:
            fig.add_trace(
                go.Histogram(
                    x=scores,
                    nbinsx=20,
                    marker_color=CATEGORICAL_COLORS[positions.index((row, col))],
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    # Validity rate bar chart
    valid = sum(1 for t in transcriptions if t.is_valid is True)
    invalid = sum(1 for t in transcriptions if t.is_valid is False)
    if valid + invalid > 0:
        fig.add_trace(
            go.Bar(
                x=["Valid", "Invalid"],
                y=[valid, invalid],
                marker_color=[CATEGORICAL_COLORS[2], CATEGORICAL_COLORS[3]],
                showlegend=False,
            ),
            row=2,
            col=3,
        )

    fig.update_layout(
        height=600,
        showlegend=False,
        template="plotly_white",
        title_text="ASR Transcription Quality Score Distributions",
    )

    if threshold is not None:
        _add_threshold_line(fig, threshold, row=1, col=1)

    return fig


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


# ---------------------------------------------------------------------------
# New charts
# ---------------------------------------------------------------------------


def create_correlation_heatmap(qa_pairs: list[QAPairRow]) -> go.Figure:
    """Create a Pearson correlation matrix across validation criteria and confidence.

    Annotated with r-values. Diverging blue-white-red color scale.

    Args:
        qa_pairs: List of QAPairRow objects to visualize.

    Returns:
        Plotly Figure object.
    """
    fields = [
        "faithfulness",
        "bloom_calibration",
        "informativeness",
        "self_containedness",
        "confidence",
    ]
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


def create_quality_radar_chart(
    qa_pairs: list[QAPairRow],
    transcriptions: list[TranscriptionRow],
    pipeline_ids: list[str],
) -> go.Figure:
    """Create a radar chart comparing mean quality metrics across runs.

    One trace per run. Axes: mean of each validation criterion + mean
    transcription quality.

    Args:
        qa_pairs: List of QAPairRow objects.
        transcriptions: List of TranscriptionRow objects.
        pipeline_ids: List of pipeline IDs to include.

    Returns:
        Plotly Figure object.
    """
    criteria = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"]
    theta = [c.replace("_", " ").title() for c in criteria] + ["Transcription Quality"]

    fig = go.Figure()

    for i, pid in enumerate(pipeline_ids):
        run_qa = [qa for qa in qa_pairs if qa.pipeline_id == pid]
        run_trans = [t for t in transcriptions if t.pipeline_id == pid]

        values = []
        for criterion in criteria:
            scores = [getattr(qa, criterion) for qa in run_qa if getattr(qa, criterion) is not None]
            values.append(sum(scores) / len(scores) if scores else 0.0)

        # Mean transcription quality
        tq_scores = [t.overall_quality for t in run_trans if t.overall_quality is not None]
        values.append(sum(tq_scores) / len(tq_scores) if tq_scores else 0.0)

        fig.add_trace(
            go.Scatterpolar(
                r=[*values, values[0]],  # close the polygon
                theta=[*theta, theta[0]],
                fill="toself",
                name=pid,
                marker_color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)],
            )
        )

    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        title="Mean Quality Profile per Pipeline Run",
        template="plotly_white",
        height=500,
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


def create_participant_breakdown_chart(
    qa_pairs: list[QAPairRow],
    transcriptions: list[TranscriptionRow],
) -> go.Figure:
    """Create a grouped bar chart showing document and QA pair counts by participant.

    Args:
        qa_pairs: List of QAPairRow objects.
        transcriptions: List of TranscriptionRow objects.

    Returns:
        Plotly Figure object.
    """
    doc_counts: dict[str, int] = defaultdict(int)
    qa_counts: dict[str, int] = defaultdict(int)

    for t in transcriptions:
        name = t.participant_name or "Unknown"
        doc_counts[name] += 1

    for qa in qa_pairs:
        name = qa.participant_name or "Unknown"
        qa_counts[name] += 1

    participants = sorted(set(doc_counts) | set(qa_counts))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Documents",
            x=participants,
            y=[doc_counts.get(p, 0) for p in participants],
            marker_color=CATEGORICAL_COLORS[0],
        )
    )
    fig.add_trace(
        go.Bar(
            name="QA Pairs",
            x=participants,
            y=[qa_counts.get(p, 0) for p in participants],
            marker_color=CATEGORICAL_COLORS[1],
        )
    )

    fig.update_layout(
        barmode="group",
        title="Documents & QA Pairs per Participant",
        xaxis_title="Participant",
        yaxis_title="Count",
        template="plotly_white",
        height=450,
    )

    return fig


def create_location_treemap(transcriptions: list[TranscriptionRow]) -> go.Figure:
    """Create a treemap showing location > participant > document count hierarchy.

    Args:
        transcriptions: List of TranscriptionRow objects.

    Returns:
        Plotly Figure object.
    """
    labels = ["All"]
    parents = [""]
    values = [0]

    # Build hierarchy: location > participant
    loc_part: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in transcriptions:
        loc = t.location or "Unknown"
        part = t.participant_name or "Unknown"
        loc_part[loc][part] += 1

    for loc, parts in sorted(loc_part.items()):
        labels.append(loc)
        parents.append("All")
        values.append(0)

        for part, count in sorted(parts.items()):
            label = f"{part} ({loc})"
            labels.append(label)
            parents.append(loc)
            values.append(count)

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker_colorscale="Blues",
        )
    )

    fig.update_layout(
        title="Source Hierarchy: Location > Participant > Document Count",
        template="plotly_white",
        height=500,
    )

    return fig


def create_bloom_validation_heatmap(
    qa_pairs: list[QAPairRow],
    threshold: float | None = None,
) -> go.Figure:
    """Create a heatmap of mean validation scores by Bloom level and criterion.

    Rows: Bloom levels, Cols: validation criteria. Cells annotated with mean +/- std.
    When threshold is provided, cells with mean score below threshold are marked with ⚠.

    Args:
        qa_pairs: List of QAPairRow objects to visualize.
        threshold: Optional threshold; cells with mean below this value are flagged.

    Returns:
        Plotly Figure object.
    """
    bloom_levels = ["remember", "understand", "analyze", "evaluate"]
    criteria = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"]
    criteria_labels = [c.replace("_", " ").title() for c in criteria]

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
                if threshold is not None and mean < threshold:
                    cell_text = f"⚠ {cell_text}"
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
