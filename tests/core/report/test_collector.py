"""Tests for results collector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from gtranscriber.core.report.collector import ResultsCollector, RunReport
from gtranscriber.schemas import EnrichedRecord, PipelineMetadata


@pytest.fixture
def sample_results_dir(tmp_path: Path) -> Path:
    """Create a sample results directory structure for testing.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the sample results directory.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create a sample pipeline run
    pipeline_dir = results_dir / "test_pipeline_001"
    pipeline_dir.mkdir()

    # Create pipeline.json
    pipeline = PipelineMetadata(
        pipeline_id="test_pipeline_001",
        steps_run=["transcription", "cep"],
    )
    pipeline.save(pipeline_dir / "pipeline.json")

    # Create transcription step
    transcription_dir = pipeline_dir / "transcription"
    transcription_dir.mkdir()
    outputs_dir = transcription_dir / "outputs"
    outputs_dir.mkdir()

    # Create a sample transcription record
    record = EnrichedRecord(
        gdrive_id="test123",
        name="test.mp3",
        mimeType="audio/mpeg",
        parents=["parent_folder"],
        webContentLink="https://drive.google.com/test",
        transcription_text="This is a test transcription.",
        detected_language="pt",
        language_probability=0.95,
        model_id="openai/whisper-large-v3",
        compute_device="cpu",
        processing_duration_sec=10.5,
        transcription_status="completed",
    )
    (outputs_dir / "test_transcription.json").write_text(record.model_dump_json())

    return results_dir


class TestResultsCollector:
    """Tests for ResultsCollector class."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test ResultsCollector initialization."""
        collector = ResultsCollector(tmp_path)
        assert collector.results_dir == tmp_path

    def test_discover_runs_empty(self, tmp_path: Path) -> None:
        """Test discover_runs with empty directory."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        collector = ResultsCollector(results_dir)

        runs = collector.discover_runs()
        assert runs == []

    def test_discover_runs_with_data(self, sample_results_dir: Path) -> None:
        """Test discover_runs with sample data."""
        collector = ResultsCollector(sample_results_dir)

        runs = collector.discover_runs()
        assert len(runs) == 1
        assert "test_pipeline_001" in runs

    def test_load_run_not_found(self, tmp_path: Path) -> None:
        """Test load_run with non-existent run."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        collector = ResultsCollector(results_dir)

        with pytest.raises(FileNotFoundError):
            collector.load_run("nonexistent_run")

    def test_load_run_with_data(self, sample_results_dir: Path) -> None:
        """Test load_run with sample data."""
        collector = ResultsCollector(sample_results_dir)

        report = collector.load_run("test_pipeline_001")
        assert report.pipeline_id == "test_pipeline_001"
        assert report.pipeline is not None
        assert report.pipeline.pipeline_id == "test_pipeline_001"
        assert len(report.transcription_records) == 1
        assert report.transcription_records[0].name == "test.mp3"

    def test_load_all_runs(self, sample_results_dir: Path) -> None:
        """Test load_all_runs."""
        collector = ResultsCollector(sample_results_dir)

        reports = collector.load_all_runs()
        assert len(reports) == 1
        assert reports[0].pipeline_id == "test_pipeline_001"

    def test_load_all_runs_empty(self, tmp_path: Path) -> None:
        """Test load_all_runs with empty directory."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        collector = ResultsCollector(results_dir)

        reports = collector.load_all_runs()
        assert reports == []


class TestRunReport:
    """Tests for RunReport model."""

    def test_initialization(self) -> None:
        """Test RunReport initialization."""
        report = RunReport(pipeline_id="test_001")
        assert report.pipeline_id == "test_001"
        assert report.pipeline is None
        assert report.transcription_records == []
        assert report.cep_records == []

    def test_with_data(self) -> None:
        """Test RunReport with data."""
        pipeline = PipelineMetadata(
            pipeline_id="test_001",
            steps_run=["transcription"],
        )
        record = EnrichedRecord(
            gdrive_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["parent"],
            webContentLink="https://drive.google.com/test",
            transcription_text="Test",
            detected_language="pt",
            language_probability=0.95,
            model_id="whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=10.0,
            transcription_status="completed",
        )

        report = RunReport(
            pipeline_id="test_001",
            pipeline=pipeline,
            transcription_records=[record],
        )

        assert report.pipeline_id == "test_001"
        assert report.pipeline is not None
        assert len(report.transcription_records) == 1
        assert report.transcription_records[0].name == "test.mp3"
