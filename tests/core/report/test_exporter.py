"""Tests for PNG chart exporter."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from arandu.core.report.dataset import (
    QAPairRow,
    ReportDataset,
    RunSummaryRow,
    TranscriptionRow,
)
from arandu.core.report.exporter import _consensus_threshold, export_charts_as_png

# All chart function names referenced by exporter.py
_CHART_FUNCTIONS = [
    "create_pipeline_overview_chart",
    "create_bloom_distribution_chart",
    "create_validation_scores_chart",
    "create_confidence_distribution_chart",
    "create_transcription_quality_chart",
    "create_multihop_chart",
    "create_correlation_heatmap",
    "create_quality_radar_chart",
    "create_parallel_coordinates_chart",
    "create_run_timeline_chart",
    "create_participant_breakdown_chart",
    "create_location_treemap",
    "create_bloom_validation_heatmap",
    "create_location_quality_chart",
]


def _make_dataset() -> ReportDataset:
    """Create a minimal ReportDataset for testing."""
    return ReportDataset(
        qa_pairs=[
            QAPairRow(
                pipeline_id="run_001",
                source_filename="test.mp3",
                bloom_level="analyze",
                confidence=0.9,
                faithfulness=0.85,
                bloom_calibration=0.8,
                informativeness=0.7,
                self_containedness=0.9,
                overall_score=0.81,
                is_valid=True,
            ),
        ],
        transcriptions=[
            TranscriptionRow(
                pipeline_id="run_001",
                source_filename="test.mp3",
                overall_quality=0.85,
                is_valid=True,
            ),
        ],
        runs=[
            RunSummaryRow(
                pipeline_id="run_001",
                steps_run=["transcription", "cep"],
                status="completed",
                success_rate=95.0,
                duration_seconds=120.0,
                completed_items=10,
                total_items=10,
            ),
        ],
        generated_at="2025-01-01T00:00:00Z",
    )


def _setup_mock_charts(mock_charts: MagicMock, mock_fig: MagicMock) -> None:
    """Configure all chart functions on the mock to return mock_fig."""
    for name in _CHART_FUNCTIONS:
        getattr(mock_charts, name).return_value = mock_fig


class TestExportChartsAsPng:
    """Tests for export_charts_as_png function."""

    @patch("arandu.core.report.exporter.charts")
    def test_creates_output_directory(self, mock_charts: MagicMock, tmp_path: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "figures"
        mock_fig = MagicMock()
        _setup_mock_charts(mock_charts, mock_fig)

        dataset = _make_dataset()
        export_charts_as_png(dataset, output_dir)

        assert output_dir.exists()

    @patch("arandu.core.report.exporter.charts")
    def test_returns_generated_paths(self, mock_charts: MagicMock, tmp_path: Path) -> None:
        """Test that generated file paths are returned."""
        output_dir = tmp_path / "figures"
        mock_fig = MagicMock()
        _setup_mock_charts(mock_charts, mock_fig)

        dataset = _make_dataset()
        result = export_charts_as_png(dataset, output_dir)

        assert isinstance(result, list)
        assert len(result) == 14
        assert mock_fig.write_image.call_count == 14

    @patch("arandu.core.report.exporter.charts")
    def test_handles_chart_failure(self, mock_charts: MagicMock, tmp_path: Path) -> None:
        """Test that individual chart failures don't stop the export."""
        output_dir = tmp_path / "figures"

        for name in _CHART_FUNCTIONS:
            getattr(mock_charts, name).side_effect = ValueError("test error")

        dataset = _make_dataset()
        result = export_charts_as_png(dataset, output_dir)

        assert result == []


class TestConsensusThreshold:
    """Tests for _consensus_threshold helper."""

    def test_single_value(self) -> None:
        """Single non-None value is returned as-is."""
        assert _consensus_threshold([0.7]) == 0.7

    def test_all_agree(self) -> None:
        """All runs share the same threshold — it is returned."""
        assert _consensus_threshold([0.7, 0.7, 0.7]) == 0.7

    def test_disagreement_returns_none(self) -> None:
        """Runs with different thresholds return None to avoid silent misapplication."""
        assert _consensus_threshold([0.6, 0.7]) is None

    def test_all_none(self) -> None:
        """No configured thresholds — returns None."""
        assert _consensus_threshold([None, None]) is None

    def test_empty_list(self) -> None:
        """Empty list returns None."""
        assert _consensus_threshold([]) is None

    def test_partial_none_all_agree(self) -> None:
        """Runs where some lack config but the rest agree — returns the shared value."""
        assert _consensus_threshold([None, 0.7, 0.7]) == 0.7

    def test_partial_none_disagree(self) -> None:
        """Mixed None with differing values — returns None."""
        assert _consensus_threshold([None, 0.6, 0.7]) is None
