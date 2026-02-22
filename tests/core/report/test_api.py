"""Tests for FastAPI route handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from gtranscriber.core.report.api import create_app, get_report_service
from gtranscriber.core.report.api_schemas import (
    FunnelData,
    FunnelStage,
    PaginatedResponse,
    QAPairDetail,
    TranscriptionDetail,
)
from gtranscriber.core.report.dataset import (
    QAPairRow,
    RunSummaryRow,
    TranscriptionRow,
)
from gtranscriber.core.report.service import ReportService

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service() -> MagicMock:
    """Return a MagicMock ReportService."""
    return MagicMock(spec=ReportService)


@pytest.fixture
def client(tmp_path: Path, mock_service: MagicMock) -> object:
    """Return a FastAPI TestClient with the service dependency overridden."""
    from fastapi.testclient import TestClient

    app = create_app(tmp_path)
    app.dependency_overrides[get_report_service] = lambda: mock_service
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/runs
# ---------------------------------------------------------------------------


class TestListRunsEndpoint:
    """Tests for GET /api/runs."""

    def test_list_runs_endpoint(self, client: object, mock_service: MagicMock) -> None:
        """GET /api/runs returns 200 with run list."""
        mock_service.list_runs.return_value = [RunSummaryRow(pipeline_id="pipe_001")]
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["pipeline_id"] == "pipe_001"


class TestGetRunDetailEndpoint:
    """Tests for GET /api/runs/{pipeline_id}."""

    def test_get_run_detail_endpoint(self, client: object, mock_service: MagicMock) -> None:
        """GET /api/runs/{id} returns 200."""
        mock_service.get_run_summary.return_value = RunSummaryRow(pipeline_id="pipe_001")
        resp = client.get("/api/runs/pipe_001")
        assert resp.status_code == 200
        assert resp.json()["pipeline_id"] == "pipe_001"

    def test_get_run_detail_not_found(self, client: object, mock_service: MagicMock) -> None:
        """GET /api/runs/nonexistent returns 404."""
        mock_service.get_run_summary.side_effect = KeyError("not found")
        resp = client.get("/api/runs/nonexistent")
        assert resp.status_code == 404


class TestGetRunConfigEndpoint:
    """Tests for GET /api/runs/{pipeline_id}/config."""

    def test_get_run_config_endpoint(self, client: object, mock_service: MagicMock) -> None:
        """GET /api/runs/{id}/config returns config."""
        from gtranscriber.core.report.api_schemas import RunConfigResponse

        mock_service.get_run_config.return_value = RunConfigResponse(pipeline_id="pipe_001")
        resp = client.get("/api/runs/pipe_001/config")
        assert resp.status_code == 200
        assert resp.json()["pipeline_id"] == "pipe_001"


# ---------------------------------------------------------------------------
# /api/qa
# ---------------------------------------------------------------------------


class TestListQAPairsEndpoint:
    """Tests for GET /api/qa."""

    def test_list_qa_pairs_endpoint(self, client: object, mock_service: MagicMock) -> None:
        """GET /api/qa returns paginated results."""
        mock_service.list_qa_pairs.return_value = PaginatedResponse(
            items=[QAPairRow(pipeline_id="pipe_001", source_filename="audio.mp3")],
            total=1,
            page=1,
            per_page=25,
            total_pages=1,
        )
        resp = client.get("/api/qa")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["pipeline_id"] == "pipe_001"

    def test_list_qa_pairs_with_filters(self, client: object, mock_service: MagicMock) -> None:
        """GET /api/qa?pipeline=X&is_valid=true passes filters to service."""
        mock_service.list_qa_pairs.return_value = PaginatedResponse(
            items=[], total=0, page=1, per_page=25, total_pages=1
        )
        resp = client.get("/api/qa?pipeline=pipe_001&is_valid=true")
        assert resp.status_code == 200
        mock_service.list_qa_pairs.assert_called_once()
        filters = mock_service.list_qa_pairs.call_args[0][0]
        assert filters.pipeline == "pipe_001"
        assert filters.is_valid is True

    def test_invalid_filter_params(self, client: object, mock_service: MagicMock) -> None:
        """Invalid sort_order returns 422."""
        resp = client.get("/api/qa?sort_order=invalid")
        assert resp.status_code == 422


class TestGetQADetailEndpoint:
    """Tests for GET /api/qa/{pipeline}/{file}/{idx}."""

    def test_get_qa_detail_endpoint(self, client: object, mock_service: MagicMock) -> None:
        """GET /api/qa/{pipeline}/{file}/{idx} returns detail."""
        mock_service.get_qa_detail.return_value = QAPairDetail(
            summary=QAPairRow(pipeline_id="pipe_001", source_filename="audio.mp3"),
            question="Q?",
            answer="A.",
            context="C",
        )
        resp = client.get("/api/qa/pipe_001/audio.mp3/0")
        assert resp.status_code == 200
        assert resp.json()["question"] == "Q?"

    def test_get_qa_detail_not_found(self, client: object, mock_service: MagicMock) -> None:
        """GET /api/qa/{pipeline}/{file}/{idx} returns 404 when not found."""
        mock_service.get_qa_detail.side_effect = KeyError("not found")
        resp = client.get("/api/qa/pipe_001/missing.mp3/0")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/transcriptions
# ---------------------------------------------------------------------------


class TestListTranscriptionsEndpoint:
    """Tests for GET /api/transcriptions."""

    def test_list_transcriptions_endpoint(self, client: object, mock_service: MagicMock) -> None:
        """GET /api/transcriptions returns paginated results."""
        mock_service.list_transcriptions.return_value = PaginatedResponse(
            items=[TranscriptionRow(pipeline_id="pipe_001", source_filename="audio.mp3")],
            total=1,
            page=1,
            per_page=25,
            total_pages=1,
        )
        resp = client.get("/api/transcriptions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1


class TestGetTranscriptionDetailEndpoint:
    """Tests for GET /api/transcriptions/{pipeline}/{file}."""

    def test_get_transcription_detail_endpoint(
        self, client: object, mock_service: MagicMock
    ) -> None:
        """GET /api/transcriptions/{pipeline}/{file} returns detail."""
        mock_service.get_transcription_detail.return_value = TranscriptionDetail(
            summary=TranscriptionRow(pipeline_id="pipe_001", source_filename="audio.mp3")
        )
        resp = client.get("/api/transcriptions/pipe_001/audio.mp3")
        assert resp.status_code == 200
        assert resp.json()["summary"]["pipeline_id"] == "pipe_001"

    def test_get_transcription_detail_not_found(
        self, client: object, mock_service: MagicMock
    ) -> None:
        """GET /api/transcriptions/{pipeline}/{file} returns 404 when not found."""
        mock_service.get_transcription_detail.side_effect = KeyError("not found")
        resp = client.get("/api/transcriptions/pipe_001/missing.mp3")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/funnel
# ---------------------------------------------------------------------------


class TestGetFunnelEndpoint:
    """Tests for GET /api/funnel/{pipeline_id}."""

    def test_get_funnel_endpoint(self, client: object, mock_service: MagicMock) -> None:
        """GET /api/funnel/{id} returns funnel data."""
        mock_service.get_funnel.return_value = FunnelData(
            pipeline_id="pipe_001",
            stages=[FunnelStage(label="Total", count=10, drop_count=0)],
        )
        resp = client.get("/api/funnel/pipe_001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["pipeline_id"] == "pipe_001"
        assert body["stages"][0]["count"] == 10

    def test_get_funnel_not_found(self, client: object, mock_service: MagicMock) -> None:
        """GET /api/funnel/{id} returns 404 when not found."""
        mock_service.get_funnel.side_effect = KeyError("not found")
        resp = client.get("/api/funnel/nonexistent")
        assert resp.status_code == 404
