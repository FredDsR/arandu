"""Interactive Plotly chart builders for HTML reports.

Creates responsive, interactive visualizations for embedding in HTML reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .style import CATEGORICAL_COLORS, get_bloom_color

if TYPE_CHECKING:
    from .collector import RunReport


def create_pipeline_overview_chart(reports: list[RunReport]) -> go.Figure:
    """Create a bar chart showing success rates across pipeline runs.

    Args:
        reports: List of RunReport objects to visualize.

    Returns:
        Plotly Figure object.
    """
    pipeline_ids = []
    success_rates = []
    durations = []

    for report in reports:
        if report.transcription_metadata or report.cep_metadata:
            # Prefer CEP metadata if available, otherwise use transcription
            metadata = report.cep_metadata or report.transcription_metadata
            if metadata and metadata.success_rate is not None:
                pipeline_ids.append(report.pipeline_id)
                success_rates.append(metadata.success_rate)
                durations.append(metadata.duration_seconds or 0)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Success Rate by Run", "Processing Duration (seconds)"),
    )

    # Success rate bar chart
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

    # Duration bar chart
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
        title_text="Pipeline Overview Dashboard",
    )

    return fig


def create_bloom_distribution_chart(reports: list[RunReport]) -> go.Figure:
    """Create a stacked bar chart showing Bloom's taxonomy distribution.

    Args:
        reports: List of RunReport objects to visualize.

    Returns:
        Plotly Figure object.
    """
    pipeline_ids = []
    bloom_data = {"remember": [], "understand": [], "analyze": [], "evaluate": []}

    for report in reports:
        if report.cep_records:
            pipeline_ids.append(report.pipeline_id)
            # Aggregate bloom distribution across all CEP records
            total_bloom = {"remember": 0, "understand": 0, "analyze": 0, "evaluate": 0}
            for cep_record in report.cep_records:
                for level, count in cep_record.bloom_distribution.items():
                    if level in total_bloom:
                        total_bloom[level] += count

            for level in bloom_data:
                bloom_data[level].append(total_bloom[level])

    fig = go.Figure()

    for level in ["remember", "understand", "analyze", "evaluate"]:
        fig.add_trace(
            go.Bar(
                name=level.capitalize(),
                x=pipeline_ids,
                y=bloom_data[level],
                marker_color=get_bloom_color(level),
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Bloom's Taxonomy Distribution by Run",
        xaxis_title="Pipeline ID",
        yaxis_title="Number of QA Pairs",
        template="plotly_white",
        height=450,
    )

    return fig


def create_validation_scores_boxplot(reports: list[RunReport]) -> go.Figure:
    """Create box plots showing validation score distributions.

    Args:
        reports: List of RunReport objects to visualize.

    Returns:
        Plotly Figure object.
    """
    criteria = ["faithfulness", "bloom_calibration", "informativeness", "self_containedness"]
    criterion_data = {c: [] for c in criteria}

    for report in reports:
        for cep_record in report.cep_records:
            for qa_pair in cep_record.qa_pairs:
                # Check if this is a validated pair
                if hasattr(qa_pair, "validation") and qa_pair.validation:
                    for criterion in criteria:
                        score = getattr(qa_pair.validation, criterion, None)
                        if score is not None:
                            criterion_data[criterion].append(score)

    fig = go.Figure()

    for i, criterion in enumerate(criteria):
        if criterion_data[criterion]:
            fig.add_trace(
                go.Box(
                    y=criterion_data[criterion],
                    name=criterion.replace("_", " ").title(),
                    marker_color=CATEGORICAL_COLORS[i],
                )
            )

    fig.update_layout(
        title="Validation Score Distributions by Criterion",
        yaxis_title="Score",
        template="plotly_white",
        height=450,
    )

    return fig


def create_confidence_distribution_chart(reports: list[RunReport]) -> go.Figure:
    """Create a histogram showing confidence score distribution.

    Args:
        reports: List of RunReport objects to visualize.

    Returns:
        Plotly Figure object.
    """
    confidence_scores = []

    for report in reports:
        for cep_record in report.cep_records:
            for qa_pair in cep_record.qa_pairs:
                if hasattr(qa_pair, "confidence_score") and qa_pair.confidence_score:
                    confidence_scores.append(qa_pair.confidence_score)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=confidence_scores,
            nbinsx=30,
            marker_color=CATEGORICAL_COLORS[0],
            name="Confidence Score",
        )
    )

    fig.update_layout(
        title="QA Pair Confidence Score Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
    )

    return fig


def create_transcription_quality_chart(reports: list[RunReport]) -> go.Figure:
    """Create histograms showing transcription quality score distributions.

    Args:
        reports: List of RunReport objects to visualize.

    Returns:
        Plotly Figure object.
    """
    quality_scores = {
        "overall_score": [],
        "script_match_score": [],
        "repetition_score": [],
        "segment_quality_score": [],
        "content_density_score": [],
    }

    for report in reports:
        for record in report.transcription_records:
            if record.transcription_quality:
                quality = record.transcription_quality
                quality_scores["overall_score"].append(quality.overall_score)
                quality_scores["script_match_score"].append(quality.script_match_score)
                quality_scores["repetition_score"].append(quality.repetition_score)
                quality_scores["segment_quality_score"].append(quality.segment_quality_score)
                quality_scores["content_density_score"].append(quality.content_density_score)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Overall Score",
            "Script Match",
            "Repetition",
            "Segment Quality",
            "Content Density",
            "Validity Rate",
        ),
    )

    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    score_types = [
        "overall_score",
        "script_match_score",
        "repetition_score",
        "segment_quality_score",
        "content_density_score",
    ]

    for (row, col), score_type in zip(positions, score_types, strict=False):
        if quality_scores[score_type]:
            fig.add_trace(
                go.Histogram(
                    x=quality_scores[score_type],
                    nbinsx=20,
                    marker_color=CATEGORICAL_COLORS[positions.index((row, col))],
                    name=score_type.replace("_", " ").title(),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    # Add validity rate
    valid_count = sum(1 for r in reports for rec in r.transcription_records if rec.is_valid)
    total_count = sum(len(r.transcription_records) for r in reports)
    if total_count > 0:
        fig.add_trace(
            go.Bar(
                x=["Valid", "Invalid"],
                y=[valid_count, total_count - valid_count],
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
        title_text="Transcription Quality Metrics",
    )

    return fig


def create_multihop_ratio_chart(reports: list[RunReport]) -> go.Figure:
    """Create a pie chart showing multi-hop vs single-hop question ratio.

    Args:
        reports: List of RunReport objects to visualize.

    Returns:
        Plotly Figure object.
    """
    multihop_count = 0
    single_hop_count = 0

    for report in reports:
        for cep_record in report.cep_records:
            for qa_pair in cep_record.qa_pairs:
                if hasattr(qa_pair, "is_multi_hop") and qa_pair.is_multi_hop is not None:
                    if qa_pair.is_multi_hop:
                        multihop_count += 1
                    else:
                        single_hop_count += 1

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Multi-hop", "Single-hop"],
                values=[multihop_count, single_hop_count],
                marker_colors=[CATEGORICAL_COLORS[0], CATEGORICAL_COLORS[1]],
            )
        ]
    )

    fig.update_layout(
        title="Multi-hop vs Single-hop Questions",
        template="plotly_white",
        height=400,
    )

    return fig
