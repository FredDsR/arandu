"""Tests for HTML report generation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from gtranscriber.core.report.collector import RunReport
from gtranscriber.core.report.dataset import (
    ReportDataset,
)
from gtranscriber.core.report.generator import generate_html_report


def _make_minimal_reports() -> list[RunReport]:
    """Create a minimal list of RunReport objects for testing."""
    return [RunReport(pipeline_id="run_001")]


class TestGenerateHtmlReport:
    """Tests for generate_html_report function."""

    @patch("gtranscriber.core.report.generator.build_dataset")
    def test_creates_file(self, mock_build: MagicMock, tmp_path: Path) -> None:
        """Test that an HTML file is created at the output path."""
        mock_build.return_value = ReportDataset(generated_at="2025-01-01T00:00:00Z")
        output = tmp_path / "report.html"

        generate_html_report(_make_minimal_reports(), output, self_contained=False)

        assert output.exists()
        assert output.stat().st_size > 0

    @patch("gtranscriber.core.report.generator.build_dataset")
    def test_contains_run_selector(self, mock_build: MagicMock, tmp_path: Path) -> None:
        """Test that output HTML contains the run selector for API-driven loading."""
        mock_build.return_value = ReportDataset(generated_at="2025-01-01T00:00:00Z")
        output = tmp_path / "report.html"

        generate_html_report(_make_minimal_reports(), output, self_contained=False)

        html = output.read_text(encoding="utf-8")
        assert "run-selector" in html
        assert "/api/runs" in html

    @patch("gtranscriber.core.report.generator.build_dataset")
    def test_contains_template_structure(self, mock_build: MagicMock, tmp_path: Path) -> None:
        """Test that output HTML contains key template elements."""
        mock_build.return_value = ReportDataset(generated_at="2025-01-01T00:00:00Z")
        output = tmp_path / "report.html"

        generate_html_report(_make_minimal_reports(), output, self_contained=False)

        html = output.read_text(encoding="utf-8")
        assert "filter-bar" in html
        assert "tab-nav" in html
        assert "G-Transcriber Pipeline Report" in html
        # 6 tabs
        assert 'data-tab="overview"' in html
        assert 'data-tab="qa"' in html
        assert 'data-tab="transcriptions"' in html
        assert 'data-tab="source"' in html
        assert 'data-tab="config"' in html
        assert 'data-tab="compare"' in html
        # sub-tabs
        assert "sub-tab-nav" in html
        assert "subtab-qa-charts" in html
        assert "subtab-trans-charts" in html
        # new filter controls
        assert "filter-validity" in html
        assert "filter-min-score" in html
        assert "filter-search" in html

    @patch("gtranscriber.core.report.generator.build_dataset")
    def test_cdn_fallback_without_self_contained(
        self, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        """Test that CDN script tag is used when self_contained=False."""
        mock_build.return_value = ReportDataset(generated_at="2025-01-01T00:00:00Z")
        output = tmp_path / "report.html"

        generate_html_report(_make_minimal_reports(), output, self_contained=False)

        html = output.read_text(encoding="utf-8")
        assert "cdn.plot.ly/plotly-latest.min.js" in html

    @patch("plotly.offline.get_plotlyjs", return_value="/* mock plotly.js */")
    @patch("gtranscriber.core.report.generator.build_dataset")
    def test_self_contained_embeds_plotly(
        self, mock_build: MagicMock, mock_plotly: MagicMock, tmp_path: Path
    ) -> None:
        """Test that Plotly.js is embedded inline when self_contained=True."""
        mock_build.return_value = ReportDataset(generated_at="2025-01-01T00:00:00Z")
        output = tmp_path / "report.html"

        generate_html_report(_make_minimal_reports(), output, self_contained=True)

        html = output.read_text(encoding="utf-8")
        assert "/* mock plotly.js */" in html
        assert "cdn.plot.ly" not in html

    @patch("gtranscriber.core.report.generator.build_dataset")
    def test_creates_parent_directories(self, mock_build: MagicMock, tmp_path: Path) -> None:
        """Test that parent directories are created if they don't exist."""
        mock_build.return_value = ReportDataset(generated_at="2025-01-01T00:00:00Z")
        output = tmp_path / "nested" / "dirs" / "report.html"

        generate_html_report(_make_minimal_reports(), output, self_contained=False)

        assert output.exists()
        assert output.parent.exists()

    @patch("gtranscriber.core.report.generator.build_dataset")
    def test_contains_timestamp(self, mock_build: MagicMock, tmp_path: Path) -> None:
        """Test that the generated report contains a timestamp."""
        mock_build.return_value = ReportDataset(generated_at="2025-01-01T00:00:00Z")
        output = tmp_path / "report.html"

        generate_html_report(_make_minimal_reports(), output, self_contained=False)

        html = output.read_text(encoding="utf-8")
        assert "UTC" in html
