"""Tests for results collector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from arandu.core.report.collector import ResultsCollector, RunReport
from arandu.schemas import (
    EnrichedRecord,
    PipelineMetadata,
    PipelineType,
    QARecordCEP,
    SourceMetadata,
)
from tests.core.report.helpers import make_run_metadata


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


@pytest.fixture
def extended_results_dir(tmp_path: Path) -> Path:
    """Create an extended results directory with run_metadata.json and CEP outputs.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the extended results directory.
    """

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    pipeline_dir = results_dir / "test_pipeline_001"
    pipeline_dir.mkdir()

    pipeline = PipelineMetadata(
        pipeline_id="test_pipeline_001",
        steps_run=["transcription", "cep"],
    )
    pipeline.save(pipeline_dir / "pipeline.json")

    # Transcription step with run_metadata.json
    transcription_dir = pipeline_dir / "transcription"
    transcription_dir.mkdir()
    trans_outputs = transcription_dir / "outputs"
    trans_outputs.mkdir()

    transcription_meta = make_run_metadata(
        pipeline_type=PipelineType.TRANSCRIPTION,
        config_values={"model_id": "openai/whisper-large-v3", "quality_threshold": 0.65},
        output_directory=str(transcription_dir),
    )
    transcription_meta.save(transcription_dir / "run_metadata.json")

    trans_record = EnrichedRecord(
        gdrive_id="test123",
        name="audio_sample.mp3",
        mimeType="audio/mpeg",
        parents=["parent_folder"],
        webContentLink="https://drive.google.com/test",
        transcription_text="Transcription content.",
        detected_language="pt",
        language_probability=0.95,
        model_id="openai/whisper-large-v3",
        compute_device="cpu",
        processing_duration_sec=10.5,
        transcription_status="completed",
    )
    (trans_outputs / "audio_sample_transcription.json").write_text(trans_record.model_dump_json())

    # CEP step with run_metadata.json and CEP outputs
    cep_dir = pipeline_dir / "cep"
    cep_dir.mkdir()
    cep_outputs = cep_dir / "outputs"
    cep_outputs.mkdir()

    cep_meta = make_run_metadata(
        pipeline_type=PipelineType.CEP,
        config_values={
            "model_id": "gpt-4o",
            "validator_model_id": "gpt-4o-mini",
            "provider": "openai",
            "validation_threshold": 0.75,
        },
        output_directory=str(cep_dir),
    )
    cep_meta.save(cep_dir / "run_metadata.json")

    cep_record = QARecordCEP(
        source_gdrive_id="gdrive_abc",
        source_filename="audio_sample.mp3",
        source_metadata=SourceMetadata(
            participant_name="Maria",
            location="Pelotas",
            recording_date="2024-05-15",
        ),
        transcription_text="Transcription content.",
        qa_pairs=[],
        model_id="gpt-4o",
        validator_model_id="gpt-4o-mini",
        provider="openai",
        total_pairs=0,
        bloom_distribution={},
    )
    (cep_outputs / "audio_sample_cep_qa.json").write_text(cep_record.model_dump_json())

    return results_dir


class TestResultsCollectorOnDemandLoading:
    """Tests for on-demand loading methods of ResultsCollector."""

    def test_load_run_config_for_step(self, extended_results_dir: Path) -> None:
        """Verify config loaded for a specific step."""
        collector = ResultsCollector(extended_results_dir)

        cep_config = collector.load_run_config("test_pipeline_001", "cep")
        assert cep_config is not None
        assert cep_config.config_values["model_id"] == "gpt-4o"
        assert cep_config.config_values["validation_threshold"] == 0.75

        trans_config = collector.load_run_config("test_pipeline_001", "transcription")
        assert trans_config is not None
        assert trans_config.config_values["model_id"] == "openai/whisper-large-v3"

    def test_load_run_config_missing(self, tmp_path: Path) -> None:
        """Verify None return when no run_metadata.json exists for the step."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "empty_run").mkdir()

        collector = ResultsCollector(results_dir)
        config = collector.load_run_config("empty_run", "cep")

        assert config is None

    def test_load_qa_record_found(self, extended_results_dir: Path) -> None:
        """Verify loading a specific QA record by source filename."""
        collector = ResultsCollector(extended_results_dir)
        record = collector.load_qa_record("test_pipeline_001", "audio_sample.mp3")

        assert record is not None
        assert record.source_filename == "audio_sample.mp3"

    def test_load_qa_record_not_found(self, extended_results_dir: Path) -> None:
        """Verify None for missing QA record."""
        collector = ResultsCollector(extended_results_dir)
        record = collector.load_qa_record("test_pipeline_001", "nonexistent.mp3")

        assert record is None

    def test_load_transcription_record_found(self, extended_results_dir: Path) -> None:
        """Verify loading a specific transcription record by source filename."""
        collector = ResultsCollector(extended_results_dir)
        record = collector.load_transcription_record("test_pipeline_001", "audio_sample.mp3")

        assert record is not None
        assert record.name == "audio_sample.mp3"

    def test_load_transcription_record_not_found(self, extended_results_dir: Path) -> None:
        """Verify None for missing transcription record."""
        collector = ResultsCollector(extended_results_dir)
        record = collector.load_transcription_record("test_pipeline_001", "nonexistent.mp3")

        assert record is None

    def test_load_all_run_configs(self, extended_results_dir: Path) -> None:
        """Verify loading configs from both transcription and CEP steps."""
        collector = ResultsCollector(extended_results_dir)
        configs = collector.load_all_run_configs("test_pipeline_001")

        assert "transcription" in configs
        assert "cep" in configs
        assert configs["transcription"].config_values["model_id"] == "openai/whisper-large-v3"
        assert configs["cep"].config_values["model_id"] == "gpt-4o"

    def test_load_all_run_configs_empty(self, tmp_path: Path) -> None:
        """Verify empty dict when no configs exist."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "no_steps_run").mkdir()

        collector = ResultsCollector(results_dir)
        configs = collector.load_all_run_configs("no_steps_run")

        assert configs == {}
