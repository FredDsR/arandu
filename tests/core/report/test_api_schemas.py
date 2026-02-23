"""Tests for report API response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gtranscriber.core.report.api_schemas import (
    FunnelData,
    FunnelStage,
    PaginatedResponse,
    QAFilterParams,
    QAPairDetail,
    TranscriptionDetail,
    TranscriptionFilterParams,
)
from gtranscriber.core.report.dataset import QAPairRow, TranscriptionRow


class TestPaginatedResponse:
    """Tests for PaginatedResponse schema."""

    def test_paginated_response_valid(self) -> None:
        """Valid pagination fields are accepted."""
        resp = PaginatedResponse(items=["a", "b"], total=2, page=1, per_page=25, total_pages=1)
        assert resp.total == 2
        assert resp.page == 1
        assert resp.per_page == 25
        assert resp.total_pages == 1

    def test_paginated_response_invalid_page(self) -> None:
        """Page number less than 1 is rejected."""
        with pytest.raises(ValidationError):
            PaginatedResponse(items=[], total=0, page=0, per_page=25, total_pages=0)

    def test_paginated_response_invalid_per_page_over_max(self) -> None:
        """per_page greater than 250 is rejected."""
        with pytest.raises(ValidationError):
            PaginatedResponse(items=[], total=0, page=1, per_page=251, total_pages=0)

    def test_paginated_response_negative_total(self) -> None:
        """Negative total is rejected."""
        with pytest.raises(ValidationError):
            PaginatedResponse(items=[], total=-1, page=1, per_page=25, total_pages=0)


class TestQAPairDetail:
    """Tests for QAPairDetail composition schema."""

    def test_qa_pair_detail_composition(self) -> None:
        """QAPairRow is composed correctly as summary field."""
        summary = QAPairRow(
            pipeline_id="pipe_001",
            source_filename="audio.mp3",
            overall_score=0.85,
        )
        detail = QAPairDetail(
            summary=summary,
            question="What is it?",
            answer="It is X.",
            context="Some context",
        )
        assert detail.summary.pipeline_id == "pipe_001"
        assert detail.question == "What is it?"
        assert detail.answer == "It is X."
        assert detail.context == "Some context"
        assert detail.reasoning_trace is None
        assert detail.validation_rationale is None

    def test_qa_pair_detail_optional_fields(self) -> None:
        """Optional fields default to None."""
        summary = QAPairRow(pipeline_id="p", source_filename="f.mp3")
        detail = QAPairDetail(summary=summary, question="Q", answer="A", context="C")
        assert detail.tacit_inference is None
        assert detail.generation_thinking is None


class TestTranscriptionDetail:
    """Tests for TranscriptionDetail composition schema."""

    def test_transcription_detail_composition(self) -> None:
        """TranscriptionRow is composed correctly as summary field."""
        summary = TranscriptionRow(
            pipeline_id="pipe_001",
            source_filename="audio.mp3",
            overall_quality=0.9,
        )
        detail = TranscriptionDetail(summary=summary)
        assert detail.summary.pipeline_id == "pipe_001"
        assert detail.issues_detected == []
        assert detail.segment_count == 0
        assert detail.total_duration_sec is None
        assert detail.transcription_text_preview == ""

    def test_transcription_detail_with_issues(self) -> None:
        """issues_detected list is stored correctly."""
        summary = TranscriptionRow(pipeline_id="p", source_filename="f.mp3")
        detail = TranscriptionDetail(
            summary=summary,
            issues_detected=["high repetition", "low density"],
            quality_rationale="Needs review",
        )
        assert len(detail.issues_detected) == 2
        assert detail.quality_rationale == "Needs review"


class TestQAFilterParams:
    """Tests for QAFilterParams defaults and validation."""

    def test_qa_filter_params_defaults(self) -> None:
        """Default values are set correctly."""
        params = QAFilterParams()
        assert params.page == 1
        assert params.per_page == 25
        assert params.sort_by == "overall_score"
        assert params.sort_order == "desc"
        assert params.pipeline is None
        assert params.is_valid is None

    def test_qa_filter_params_invalid_sort_order(self) -> None:
        """Only 'asc' and 'desc' are valid sort orders."""
        with pytest.raises(ValidationError):
            QAFilterParams(sort_order="random")

    def test_qa_filter_params_valid_sort_orders(self) -> None:
        """Both 'asc' and 'desc' are accepted."""
        assert QAFilterParams(sort_order="asc").sort_order == "asc"
        assert QAFilterParams(sort_order="desc").sort_order == "desc"

    def test_qa_filter_params_score_range(self) -> None:
        """Score out of [0.0, 1.0] range is rejected."""
        with pytest.raises(ValidationError):
            QAFilterParams(min_score=1.5)

    def test_qa_filter_params_page_ge_1(self) -> None:
        """Page must be >= 1."""
        with pytest.raises(ValidationError):
            QAFilterParams(page=0)


class TestTranscriptionFilterParams:
    """Tests for TranscriptionFilterParams defaults and validation."""

    def test_transcription_filter_params_defaults(self) -> None:
        """Default values are set correctly."""
        params = TranscriptionFilterParams()
        assert params.page == 1
        assert params.per_page == 25
        assert params.sort_by == "overall_quality"
        assert params.sort_order == "desc"

    def test_transcription_filter_params_invalid_sort_order(self) -> None:
        """Only 'asc' and 'desc' are valid sort orders."""
        with pytest.raises(ValidationError):
            TranscriptionFilterParams(sort_order="invalid")


class TestFunnelData:
    """Tests for FunnelData and FunnelStage schemas."""

    def test_funnel_data_stages(self) -> None:
        """Stages list is correctly structured."""
        funnel = FunnelData(
            pipeline_id="pipe_001",
            stages=[
                FunnelStage(label="Total Transcriptions", count=10, drop_count=0),
                FunnelStage(label="Valid Transcriptions", count=8, drop_count=2),
            ],
        )
        assert funnel.pipeline_id == "pipe_001"
        assert len(funnel.stages) == 2
        assert funnel.stages[0].label == "Total Transcriptions"
        assert funnel.stages[0].count == 10
        assert funnel.stages[1].drop_count == 2

    def test_funnel_stage_negative_count_rejected(self) -> None:
        """Negative count is rejected."""
        with pytest.raises(ValidationError):
            FunnelStage(label="Stage", count=-1)

    def test_funnel_data_empty_stages(self) -> None:
        """Empty stages list is valid."""
        funnel = FunnelData(pipeline_id="pipe_001")
        assert funnel.stages == []
