"""Tests for chart builder functions."""

from __future__ import annotations

import plotly.graph_objects as go

from gtranscriber.core.report.charts import (
    _empty_figure,
    _pearson_r,
    create_bloom_distribution_chart,
    create_bloom_validation_heatmap,
    create_confidence_distribution_chart,
    create_correlation_heatmap,
    create_location_quality_chart,
    create_location_treemap,
    create_multihop_chart,
    create_parallel_coordinates_chart,
    create_participant_breakdown_chart,
    create_pipeline_overview_chart,
    create_quality_radar_chart,
    create_run_timeline_chart,
    create_transcription_quality_chart,
    create_validation_scores_chart,
)
from gtranscriber.core.report.dataset import QAPairRow, RunSummaryRow, TranscriptionRow


def _make_qa_pair(
    pipeline_id: str = "run_001",
    bloom_level: str = "analyze",
    confidence: float = 0.9,
    faithfulness: float = 0.85,
    bloom_calibration: float = 0.8,
    informativeness: float = 0.7,
    self_containedness: float = 0.9,
    overall_score: float = 0.81,
    is_multi_hop: bool = False,
    location: str | None = "Pelotas",
) -> QAPairRow:
    """Create a sample QAPairRow for testing."""
    return QAPairRow(
        pipeline_id=pipeline_id,
        source_filename="test.mp3",
        bloom_level=bloom_level,
        confidence=confidence,
        faithfulness=faithfulness,
        bloom_calibration=bloom_calibration,
        informativeness=informativeness,
        self_containedness=self_containedness,
        overall_score=overall_score,
        is_multi_hop=is_multi_hop,
        location=location,
        participant_name="Maria",
    )


def _make_run(
    pipeline_id: str = "run_001",
    success_rate: float | None = 95.0,
    duration_seconds: float | None = 120.0,
    total_items: int = 10,
    created_at: str | None = "2025-01-15T10:00:00",
) -> RunSummaryRow:
    """Create a sample RunSummaryRow for testing."""
    return RunSummaryRow(
        pipeline_id=pipeline_id,
        steps_run=["transcription", "cep"],
        status="completed",
        success_rate=success_rate,
        duration_seconds=duration_seconds,
        completed_items=total_items,
        total_items=total_items,
        created_at=created_at,
    )


def _make_transcription(
    pipeline_id: str = "run_001",
    overall_quality: float = 0.85,
    location: str | None = "Pelotas",
    participant_name: str | None = "Maria",
    is_valid: bool | None = True,
) -> TranscriptionRow:
    """Create a sample TranscriptionRow for testing."""
    return TranscriptionRow(
        pipeline_id=pipeline_id,
        source_filename="test.mp3",
        overall_quality=overall_quality,
        script_match=0.9,
        repetition=0.8,
        segment_quality=0.85,
        content_density=0.7,
        is_valid=is_valid,
        location=location,
        participant_name=participant_name,
    )


class TestPipelineOverviewChart:
    """Tests for create_pipeline_overview_chart."""

    def test_returns_figure(self) -> None:
        """Test that a valid figure is returned."""
        runs = [_make_run(), _make_run("run_002", success_rate=80.0)]
        fig = create_pipeline_overview_chart(runs)
        assert isinstance(fig, go.Figure)

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_pipeline_overview_chart([])
        assert isinstance(fig, go.Figure)

    def test_single_run(self) -> None:
        """Test with a single run."""
        fig = create_pipeline_overview_chart([_make_run()])
        assert isinstance(fig, go.Figure)


class TestBloomDistributionChart:
    """Tests for create_bloom_distribution_chart."""

    def test_returns_figure(self) -> None:
        """Test with multiple bloom levels."""
        qa_pairs = [
            _make_qa_pair(bloom_level="remember"),
            _make_qa_pair(bloom_level="analyze"),
            _make_qa_pair(bloom_level="evaluate"),
        ]
        fig = create_bloom_distribution_chart(qa_pairs)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # one trace per bloom level

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_bloom_distribution_chart([])
        assert isinstance(fig, go.Figure)


class TestValidationScoresChart:
    """Tests for create_validation_scores_chart."""

    def test_returns_figure(self) -> None:
        """Test with valid QA pairs."""
        qa_pairs = [_make_qa_pair(), _make_qa_pair(faithfulness=0.6)]
        fig = create_validation_scores_chart(qa_pairs)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # one violin per criterion

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_validation_scores_chart([])
        assert isinstance(fig, go.Figure)

    def test_missing_fields(self) -> None:
        """Test with QA pairs missing validation scores."""
        qa_pairs = [
            QAPairRow(pipeline_id="r1", source_filename="a.mp3"),
        ]
        fig = create_validation_scores_chart(qa_pairs)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0  # no data to plot


class TestConfidenceDistributionChart:
    """Tests for create_confidence_distribution_chart."""

    def test_returns_figure(self) -> None:
        """Test with valid data."""
        qa_pairs = [_make_qa_pair(confidence=0.9), _make_qa_pair(confidence=0.5)]
        fig = create_confidence_distribution_chart(qa_pairs)
        assert isinstance(fig, go.Figure)

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_confidence_distribution_chart([])
        assert isinstance(fig, go.Figure)


class TestTranscriptionQualityChart:
    """Tests for create_transcription_quality_chart."""

    def test_returns_figure(self) -> None:
        """Test with valid data."""
        transcriptions = [_make_transcription(), _make_transcription(overall_quality=0.7)]
        fig = create_transcription_quality_chart(transcriptions)
        assert isinstance(fig, go.Figure)

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_transcription_quality_chart([])
        assert isinstance(fig, go.Figure)


class TestMultihopChart:
    """Tests for create_multihop_chart."""

    def test_returns_figure(self) -> None:
        """Test with mixed data."""
        qa_pairs = [_make_qa_pair(is_multi_hop=True), _make_qa_pair(is_multi_hop=False)]
        fig = create_multihop_chart(qa_pairs)
        assert isinstance(fig, go.Figure)
        assert fig.data[0].x[0] == 1  # 1 multi-hop
        assert fig.data[0].x[1] == 1  # 1 single-hop

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_multihop_chart([])
        assert isinstance(fig, go.Figure)


class TestCorrelationHeatmap:
    """Tests for create_correlation_heatmap."""

    def test_returns_figure(self) -> None:
        """Test with enough data for correlation."""
        qa_pairs = [
            _make_qa_pair(confidence=0.9, faithfulness=0.8, bloom_calibration=0.7),
            _make_qa_pair(confidence=0.7, faithfulness=0.6, bloom_calibration=0.5),
            _make_qa_pair(confidence=0.5, faithfulness=0.4, bloom_calibration=0.3),
        ]
        fig = create_correlation_heatmap(qa_pairs)
        assert isinstance(fig, go.Figure)

    def test_insufficient_data(self) -> None:
        """Test with fewer than 3 data points (identity matrix)."""
        qa_pairs = [_make_qa_pair()]
        fig = create_correlation_heatmap(qa_pairs)
        assert isinstance(fig, go.Figure)

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_correlation_heatmap([])
        assert isinstance(fig, go.Figure)


class TestQualityRadarChart:
    """Tests for create_quality_radar_chart."""

    def test_returns_figure(self) -> None:
        """Test with valid data."""
        qa_pairs = [_make_qa_pair()]
        transcriptions = [_make_transcription()]
        fig = create_quality_radar_chart(qa_pairs, transcriptions, ["run_001"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # one trace per run

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_quality_radar_chart([], [], [])
        assert isinstance(fig, go.Figure)


class TestParallelCoordinatesChart:
    """Tests for create_parallel_coordinates_chart."""

    def test_returns_figure(self) -> None:
        """Test with valid data."""
        qa_pairs = [_make_qa_pair(bloom_level="analyze")]
        fig = create_parallel_coordinates_chart(qa_pairs)
        assert isinstance(fig, go.Figure)

    def test_empty_data(self) -> None:
        """Test with empty input returns empty figure annotation."""
        fig = create_parallel_coordinates_chart([])
        assert isinstance(fig, go.Figure)
        assert fig.layout.annotations[0].text == "No data available"

    def test_missing_bloom_level(self) -> None:
        """Test with QA pairs missing bloom level falls back to empty figure."""
        qa_pairs = [QAPairRow(pipeline_id="r1", source_filename="a.mp3", confidence=0.9)]
        fig = create_parallel_coordinates_chart(qa_pairs)
        assert isinstance(fig, go.Figure)


class TestRunTimelineChart:
    """Tests for create_run_timeline_chart."""

    def test_returns_figure(self) -> None:
        """Test with valid data."""
        runs = [_make_run(), _make_run("run_002", created_at="2025-02-01T10:00:00")]
        fig = create_run_timeline_chart(runs)
        assert isinstance(fig, go.Figure)

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_run_timeline_chart([])
        assert isinstance(fig, go.Figure)

    def test_missing_dates(self) -> None:
        """Test with runs missing created_at."""
        runs = [_make_run(created_at=None)]
        fig = create_run_timeline_chart(runs)
        assert isinstance(fig, go.Figure)


class TestParticipantBreakdownChart:
    """Tests for create_participant_breakdown_chart."""

    def test_returns_figure(self) -> None:
        """Test with valid data."""
        qa_pairs = [_make_qa_pair()]
        transcriptions = [_make_transcription()]
        fig = create_participant_breakdown_chart(qa_pairs, transcriptions)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # documents + QA pairs

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_participant_breakdown_chart([], [])
        assert isinstance(fig, go.Figure)


class TestLocationTreemap:
    """Tests for create_location_treemap."""

    def test_returns_figure(self) -> None:
        """Test with valid data."""
        transcriptions = [_make_transcription(), _make_transcription(location="Cangucu")]
        fig = create_location_treemap(transcriptions)
        assert isinstance(fig, go.Figure)

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_location_treemap([])
        assert isinstance(fig, go.Figure)


class TestBloomValidationHeatmap:
    """Tests for create_bloom_validation_heatmap."""

    def test_returns_figure(self) -> None:
        """Test with valid data."""
        qa_pairs = [
            _make_qa_pair(bloom_level="remember"),
            _make_qa_pair(bloom_level="analyze"),
        ]
        fig = create_bloom_validation_heatmap(qa_pairs)
        assert isinstance(fig, go.Figure)

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_bloom_validation_heatmap([])
        assert isinstance(fig, go.Figure)


class TestLocationQualityChart:
    """Tests for create_location_quality_chart."""

    def test_returns_figure(self) -> None:
        """Test with valid data."""
        qa_pairs = [
            _make_qa_pair(location="Pelotas"),
            _make_qa_pair(location="Cangucu"),
        ]
        fig = create_location_quality_chart(qa_pairs)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # one violin per location

    def test_empty_data(self) -> None:
        """Test with empty input."""
        fig = create_location_quality_chart([])
        assert isinstance(fig, go.Figure)


class TestPearsonR:
    """Tests for _pearson_r helper."""

    def test_perfect_positive_correlation(self) -> None:
        """Test with perfectly correlated sequences."""
        x = (1.0, 2.0, 3.0, 4.0, 5.0)
        y = (2.0, 4.0, 6.0, 8.0, 10.0)
        r = _pearson_r(x, y)
        assert abs(r - 1.0) < 1e-10

    def test_perfect_negative_correlation(self) -> None:
        """Test with perfectly negatively correlated sequences."""
        x = (1.0, 2.0, 3.0, 4.0, 5.0)
        y = (10.0, 8.0, 6.0, 4.0, 2.0)
        r = _pearson_r(x, y)
        assert abs(r - (-1.0)) < 1e-10

    def test_no_correlation(self) -> None:
        """Test with uncorrelated sequences."""
        x = (1.0, 2.0, 3.0, 4.0, 5.0)
        y = (1.0, -1.0, 0.0, -1.0, 1.0)
        r = _pearson_r(x, y)
        assert abs(r) < 0.1

    def test_fewer_than_two_points(self) -> None:
        """Test with n < 2 returns 0.0."""
        assert _pearson_r((1.0,), (2.0,)) == 0.0
        assert _pearson_r((), ()) == 0.0

    def test_zero_variance(self) -> None:
        """Test with constant sequence returns 0.0."""
        x = (5.0, 5.0, 5.0)
        y = (1.0, 2.0, 3.0)
        assert _pearson_r(x, y) == 0.0


class TestEmptyFigure:
    """Tests for _empty_figure helper."""

    def test_returns_annotated_figure(self) -> None:
        """Test that an annotated figure is returned."""
        fig = _empty_figure("Test Title")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Test Title"
        assert len(fig.layout.annotations) == 1
        assert fig.layout.annotations[0].text == "No data available"
