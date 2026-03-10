"""Chart builders for HTML reports and PNG export.

Re-exports all chart functions so callers can import from
``arandu.report.charts`` without knowing the internal split.
"""

from arandu.report.charts.comparison import (
    create_cross_run_comparison,
    create_location_quality_chart,
    create_parallel_coordinates_chart,
)
from arandu.report.charts.distribution import (
    create_bloom_distribution_chart,
    create_location_treemap,
    create_participant_breakdown_chart,
)
from arandu.report.charts.multihop import create_multihop_chart
from arandu.report.charts.quality import (
    create_confidence_distribution_chart,
    create_quality_radar_chart,
    create_transcription_quality_chart,
)
from arandu.report.charts.timeline import (
    create_funnel_chart,
    create_pipeline_overview_chart,
    create_run_timeline_chart,
)
from arandu.report.charts.validation import (
    create_bloom_validation_heatmap,
    create_correlation_heatmap,
    create_validation_scores_chart,
)

__all__ = [
    "create_bloom_distribution_chart",
    "create_bloom_validation_heatmap",
    "create_confidence_distribution_chart",
    "create_correlation_heatmap",
    "create_cross_run_comparison",
    "create_funnel_chart",
    "create_location_quality_chart",
    "create_location_treemap",
    "create_multihop_chart",
    "create_parallel_coordinates_chart",
    "create_participant_breakdown_chart",
    "create_pipeline_overview_chart",
    "create_quality_radar_chart",
    "create_run_timeline_chart",
    "create_transcription_quality_chart",
    "create_validation_scores_chart",
]
