"""Quality-related charts: transcription quality, radar, and confidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .style import CATEGORICAL_COLORS

if TYPE_CHECKING:
    from arandu.report.dataset import QAPairRow, TranscriptionRow


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
        from .validation import _add_threshold_line

        _add_threshold_line(fig, threshold, row=1, col=1)

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
