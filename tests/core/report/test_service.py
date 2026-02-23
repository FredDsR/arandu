"""Tests for ReportService business logic layer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gtranscriber.core.report.api_schemas import QAFilterParams, TranscriptionFilterParams
from gtranscriber.core.report.dataset import (
    QAPairRow,
    ReportDataset,
    RunSummaryRow,
    TranscriptionRow,
)
from gtranscriber.core.report.service import ReportService
from gtranscriber.schemas import (
    ConfigSnapshot,
    QARecordCEP,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dataset(
    n_qa: int = 3,
    n_trans: int = 2,
    n_runs: int = 1,
) -> ReportDataset:
    """Build a minimal ReportDataset for testing."""
    runs = [
        RunSummaryRow(
            pipeline_id=f"pipe_{i:03d}",
            valid_transcriptions=n_trans,
            invalid_transcriptions=1,
            valid_qa_pairs=n_qa,
            invalid_qa_pairs=1,
        )
        for i in range(n_runs)
    ]
    qa_pairs = [
        QAPairRow(
            pipeline_id="pipe_000",
            source_filename=f"file_{i}.mp3",
            overall_score=0.5 + i * 0.1,
            confidence=0.7 + i * 0.05,
            bloom_level="Remember",
            is_valid=i % 2 == 0,
            location="Pelotas",
            participant_name="Maria",
        )
        for i in range(n_qa)
    ]
    transcriptions = [
        TranscriptionRow(
            pipeline_id="pipe_000",
            source_filename=f"trans_{i}.mp3",
            overall_quality=0.6 + i * 0.1,
            is_valid=i % 2 == 0,
            location="Pelotas",
            participant_name="Maria",
        )
        for i in range(n_trans)
    ]
    return ReportDataset(qa_pairs=qa_pairs, transcriptions=transcriptions, runs=runs)


@pytest.fixture
def mock_collector() -> MagicMock:
    """Return a MagicMock ResultsCollector with a preset dataset."""
    collector = MagicMock()
    collector.load_all_runs.return_value = []
    return collector


@pytest.fixture
def service(mock_collector: MagicMock) -> ReportService:
    """Return a ReportService backed by a mock collector."""
    svc = ReportService(mock_collector)
    svc._dataset = _make_dataset()
    return svc


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


class TestListRuns:
    """Tests for ReportService.list_runs."""

    def test_list_runs(self, service: ReportService) -> None:
        """Returns all run summaries from the dataset."""
        runs = service.list_runs()
        assert len(runs) == 1
        assert runs[0].pipeline_id == "pipe_000"


# ---------------------------------------------------------------------------
# list_qa_pairs
# ---------------------------------------------------------------------------


class TestListQAPairs:
    """Tests for ReportService.list_qa_pairs."""

    def test_list_qa_pairs_no_filter(self, service: ReportService) -> None:
        """Returns all QA pairs paginated."""
        result = service.list_qa_pairs(QAFilterParams())
        assert result.total == 3
        assert len(result.items) == 3

    def test_list_qa_pairs_filter_by_pipeline(self, service: ReportService) -> None:
        """Filters by pipeline_id correctly."""
        result = service.list_qa_pairs(QAFilterParams(pipeline="nonexistent"))
        assert result.total == 0

        result2 = service.list_qa_pairs(QAFilterParams(pipeline="pipe_000"))
        assert result2.total == 3

    def test_list_qa_pairs_filter_by_validity(self, service: ReportService) -> None:
        """is_valid filter works (only even indices are valid in fixture)."""
        result = service.list_qa_pairs(QAFilterParams(is_valid=True))
        assert all(r.is_valid for r in result.items)

        result_invalid = service.list_qa_pairs(QAFilterParams(is_valid=False))
        assert all(not r.is_valid for r in result_invalid.items)

    def test_list_qa_pairs_filter_by_score_range(self, service: ReportService) -> None:
        """min_score / max_score filters work."""
        # Scores are 0.5, 0.6, 0.7
        result = service.list_qa_pairs(QAFilterParams(min_score=0.65))
        assert all(r.overall_score is not None and r.overall_score >= 0.65 for r in result.items)

        result2 = service.list_qa_pairs(QAFilterParams(max_score=0.55))
        assert all(r.overall_score is not None and r.overall_score <= 0.55 for r in result2.items)

    def test_list_qa_pairs_pagination(self, service: ReportService) -> None:
        """Correct page and total_pages are returned."""
        result = service.list_qa_pairs(QAFilterParams(page=1, per_page=2))
        assert len(result.items) == 2
        assert result.total == 3
        assert result.total_pages == 2

        result2 = service.list_qa_pairs(QAFilterParams(page=2, per_page=2))
        assert len(result2.items) == 1

    def test_list_qa_pairs_sort_ascending(self, service: ReportService) -> None:
        """sort_order=asc returns items in ascending order."""
        result = service.list_qa_pairs(QAFilterParams(sort_by="overall_score", sort_order="asc"))
        scores = [r.overall_score for r in result.items if r.overall_score is not None]
        assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# get_qa_detail
# ---------------------------------------------------------------------------


class TestGetQADetail:
    """Tests for ReportService.get_qa_detail."""

    def _make_qa_record(self) -> QARecordCEP:
        """Build a minimal QARecordCEP with one QA pair."""
        from gtranscriber.schemas import QAPairCEP

        pair = QAPairCEP(
            question="What is CEP?",
            answer="Cognitive Elicitation Protocol.",
            context="CEP stands for Cognitive Elicitation Protocol.",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
        )
        return QARecordCEP(
            source_gdrive_id="gid_001",
            source_filename="audio.mp3",
            transcription_text="Full transcription text here.",
            qa_pairs=[pair],
            model_id="gpt-4o",
            provider="openai",
            total_pairs=1,
            bloom_distribution={"Remember": 1},
        )

    def test_get_qa_detail_found(self, mock_collector: MagicMock) -> None:
        """Returns composed detail when record exists."""
        record = self._make_qa_record()
        mock_collector.load_qa_record.return_value = record

        svc = ReportService(mock_collector)
        svc._dataset = _make_dataset()

        detail = svc.get_qa_detail("pipe_000", "audio.mp3", 0)
        assert detail.question == "What is CEP?"
        assert detail.answer == "Cognitive Elicitation Protocol."
        assert detail.context == "Full transcription text here."
        assert detail.summary.pipeline_id == "pipe_000"

    def test_get_qa_detail_not_found(self, mock_collector: MagicMock) -> None:
        """Raises KeyError when record does not exist."""
        mock_collector.load_qa_record.return_value = None
        svc = ReportService(mock_collector)

        with pytest.raises(KeyError, match="QA record not found"):
            svc.get_qa_detail("pipe_000", "missing.mp3", 0)

    def test_get_qa_detail_index_out_of_range(self, mock_collector: MagicMock) -> None:
        """Raises KeyError when index is out of range."""
        record = self._make_qa_record()
        mock_collector.load_qa_record.return_value = record
        svc = ReportService(mock_collector)
        svc._dataset = _make_dataset()

        with pytest.raises(KeyError):
            svc.get_qa_detail("pipe_000", "audio.mp3", 99)


# ---------------------------------------------------------------------------
# get_run_config
# ---------------------------------------------------------------------------


class TestGetRunConfig:
    """Tests for ReportService.get_run_config."""

    def test_get_run_config(self, mock_collector: MagicMock) -> None:
        """Returns config with threshold metadata."""
        trans_snap = ConfigSnapshot(
            config_type="TranscriberConfig",
            config_values={"model_id": "whisper-large-v3", "quality_threshold": 0.65},
        )
        cep_snap = ConfigSnapshot(
            config_type="CEPConfig",
            config_values={"model_id": "gpt-4o", "validation_threshold": 0.75},
        )
        mock_collector.load_all_run_configs.return_value = {
            "transcription": trans_snap,
            "cep": cep_snap,
        }
        mock_collector.load_run.side_effect = FileNotFoundError

        svc = ReportService(mock_collector)
        config = svc.get_run_config("pipe_000")

        assert config.pipeline_id == "pipe_000"
        assert "transcription" in config.configs
        assert "cep" in config.configs
        assert "quality_threshold" in config.threshold_fields["transcription"]
        assert "validation_threshold" in config.threshold_fields["cep"]

    def test_get_run_config_cep_weight_thresholds(self, mock_collector: MagicMock) -> None:
        """Returns cep weight fields as thresholds when present."""
        cep_snap = ConfigSnapshot(
            config_type="CEPConfig",
            config_values={
                "model_id": "gpt-4o",
                "validation_threshold": 0.75,
                "faithfulness_weight": 0.3,
                "bloom_calibration_weight": 0.25,
                "informativeness_weight": 0.25,
                "self_containedness_weight": 0.2,
            },
        )
        mock_collector.load_all_run_configs.return_value = {"cep": cep_snap}
        mock_collector.load_run.side_effect = FileNotFoundError

        svc = ReportService(mock_collector)
        config = svc.get_run_config("pipe_000")

        cep_thresholds = config.threshold_fields.get("cep", [])
        assert "validation_threshold" in cep_thresholds
        assert "faithfulness_weight" in cep_thresholds
        assert "bloom_calibration_weight" in cep_thresholds
        assert "informativeness_weight" in cep_thresholds
        assert "self_containedness_weight" in cep_thresholds

    def test_get_run_config_no_cross_step_leakage(self, mock_collector: MagicMock) -> None:
        """Transcription step does not inherit CEP threshold fields."""
        trans_snap = ConfigSnapshot(
            config_type="TranscriberConfig",
            config_values={
                "model_id": "whisper-large-v3",
                "quality_threshold": 0.65,
                "validation_threshold": 0.7,  # present but not a transcription threshold
            },
        )
        mock_collector.load_all_run_configs.return_value = {"transcription": trans_snap}
        mock_collector.load_run.side_effect = FileNotFoundError

        svc = ReportService(mock_collector)
        config = svc.get_run_config("pipe_000")

        trans_thresholds = config.threshold_fields.get("transcription", [])
        assert "quality_threshold" in trans_thresholds
        assert "validation_threshold" not in trans_thresholds

    def test_get_run_config_missing(self, mock_collector: MagicMock) -> None:
        """Handles missing config (empty dict) gracefully."""
        mock_collector.load_all_run_configs.return_value = {}
        mock_collector.load_run.side_effect = FileNotFoundError

        svc = ReportService(mock_collector)
        config = svc.get_run_config("pipe_empty")

        assert config.configs == {}
        assert config.threshold_fields == {}


# ---------------------------------------------------------------------------
# get_funnel
# ---------------------------------------------------------------------------


class TestGetFunnel:
    """Tests for ReportService.get_funnel."""

    def test_get_funnel(self, service: ReportService) -> None:
        """Returns correct stage counts and drop-offs."""
        funnel = service.get_funnel("pipe_000")
        assert funnel.pipeline_id == "pipe_000"
        assert len(funnel.stages) == 4
        # Stage 0: Total Transcriptions = valid + invalid = 2 + 1 = 3
        assert funnel.stages[0].count == 3
        # Stage 1: Valid Transcriptions = 2, drop = 1
        assert funnel.stages[1].count == 2
        assert funnel.stages[1].drop_count == 1

    def test_get_funnel_unknown_pipeline(self, service: ReportService) -> None:
        """Raises KeyError for unknown pipeline."""
        with pytest.raises(KeyError, match="Pipeline not found"):
            service.get_funnel("nonexistent")


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    """Tests for ReportService.export_csv."""

    def test_export_csv_qa(self, service: ReportService) -> None:
        """Generates valid CSV content for QA data."""
        csv_content = service.export_csv("qa", {})
        assert "pipeline_id" in csv_content
        assert "pipe_000" in csv_content

    def test_export_csv_transcriptions(self, service: ReportService) -> None:
        """Generates valid CSV content for transcription data."""
        csv_content = service.export_csv("transcriptions", {})
        assert "pipeline_id" in csv_content
        assert "pipe_000" in csv_content

    def test_export_csv_invalid_type(self, service: ReportService) -> None:
        """Raises ValueError for unsupported data_type."""
        with pytest.raises(ValueError, match="Unsupported data_type"):
            service.export_csv("invalid_type", {})


# ---------------------------------------------------------------------------
# list_transcriptions
# ---------------------------------------------------------------------------


class TestListTranscriptions:
    """Tests for ReportService.list_transcriptions."""

    def test_list_transcriptions_no_filter(self, service: ReportService) -> None:
        """Returns all transcriptions paginated."""
        result = service.list_transcriptions(TranscriptionFilterParams())
        assert result.total == 2

    def test_list_transcriptions_filter_by_pipeline(self, service: ReportService) -> None:
        """Filters by pipeline_id correctly."""
        result = service.list_transcriptions(TranscriptionFilterParams(pipeline="nonexistent"))
        assert result.total == 0

    def test_list_transcriptions_filter_by_validity(self, service: ReportService) -> None:
        """is_valid filter works."""
        result = service.list_transcriptions(TranscriptionFilterParams(is_valid=True))
        assert all(r.is_valid for r in result.items)
