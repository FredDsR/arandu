"""Distribution charts: Bloom taxonomy, participant breakdown, location treemap."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import plotly.graph_objects as go

from .style import CATEGORICAL_COLORS, get_bloom_color

if TYPE_CHECKING:
    from arandu.report.dataset import QAPairRow, TranscriptionRow


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
