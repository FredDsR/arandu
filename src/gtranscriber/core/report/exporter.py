"""PNG export for report charts via Plotly + kaleido.

Generates publication-quality PNG files from chart builder functions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from . import charts

if TYPE_CHECKING:
    from pathlib import Path

    from .dataset import ReportDataset

logger = logging.getLogger(__name__)


def export_charts_as_png(dataset: ReportDataset, output_dir: Path) -> list[Path]:
    """Export all charts as PNG files using kaleido.

    Calls each chart builder function from charts.py with the dataset and
    writes the result as a high-resolution PNG.

    Args:
        dataset: ReportDataset containing all flattened data.
        output_dir: Directory to save PNG files.

    Returns:
        List of paths to successfully generated PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # Extract threshold values from the first run that has them
    validation_threshold: float | None = None
    quality_threshold: float | None = None
    for run in dataset.runs:
        if validation_threshold is None and run.validation_threshold is not None:
            validation_threshold = run.validation_threshold
        if quality_threshold is None and run.quality_threshold is not None:
            quality_threshold = run.quality_threshold

    chart_specs: list[tuple[str, object]] = [
        ("pipeline_overview", lambda: charts.create_pipeline_overview_chart(dataset.runs)),
        ("bloom_distribution", lambda: charts.create_bloom_distribution_chart(dataset.qa_pairs)),
        (
            "validation_scores",
            lambda: charts.create_validation_scores_chart(
                dataset.qa_pairs, threshold=validation_threshold
            ),
        ),
        (
            "confidence_distribution",
            lambda: charts.create_confidence_distribution_chart(dataset.qa_pairs),
        ),
        (
            "transcription_quality",
            lambda: charts.create_transcription_quality_chart(
                dataset.transcriptions, threshold=quality_threshold
            ),
        ),
        ("multihop", lambda: charts.create_multihop_chart(dataset.qa_pairs)),
        (
            "correlation_heatmap",
            lambda: charts.create_correlation_heatmap(dataset.qa_pairs),
        ),
        (
            "quality_radar",
            lambda: charts.create_quality_radar_chart(
                dataset.qa_pairs, dataset.transcriptions, dataset.pipeline_ids
            ),
        ),
        (
            "parallel_coordinates",
            lambda: charts.create_parallel_coordinates_chart(dataset.qa_pairs),
        ),
        ("run_timeline", lambda: charts.create_run_timeline_chart(dataset.runs)),
        (
            "participant_breakdown",
            lambda: charts.create_participant_breakdown_chart(
                dataset.qa_pairs, dataset.transcriptions
            ),
        ),
        (
            "location_treemap",
            lambda: charts.create_location_treemap(dataset.transcriptions),
        ),
        (
            "bloom_validation_heatmap",
            lambda: charts.create_bloom_validation_heatmap(
                dataset.qa_pairs, threshold=validation_threshold
            ),
        ),
        (
            "location_quality",
            lambda: charts.create_location_quality_chart(dataset.qa_pairs),
        ),
    ]

    for name, builder in chart_specs:
        try:
            fig = builder()
            path = output_dir / f"{name}.png"
            fig.write_image(str(path), format="png", width=1200, height=600, scale=2)
            generated.append(path)
        except Exception:
            logger.warning("Failed to export chart: %s", name, exc_info=True)

    failed_count = len(chart_specs) - len(generated)
    if failed_count:
        logger.info("Chart export: %d succeeded, %d failed", len(generated), failed_count)

    return generated
