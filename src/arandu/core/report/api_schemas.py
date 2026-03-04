"""Pydantic response schemas for the report REST API.

Uses composition over inheritance: detail models wrap existing row
models rather than extending them, keeping the data layer separate
from the API contract.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from .dataset import QAPairRow, RunSummaryRow, TranscriptionRow

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):  # noqa: UP046
    """Generic paginated response wrapper.

    Attributes:
        items: List of items for the current page.
        total: Total number of items matching the query.
        page: Current page number (1-indexed).
        per_page: Number of items per page.
        total_pages: Total number of pages.
    """

    items: list[T]
    total: int = Field(..., ge=0, description="Total matching items")
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, le=250, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total pages")


class QAPairDetail(BaseModel):
    """Full QA pair detail with text content and validation rationale.

    Composes QAPairRow for summary data plus text fields loaded on demand.
    """

    summary: QAPairRow = Field(..., description="Summary metrics and metadata")
    question: str = Field(..., description="Full question text")
    answer: str = Field(..., description="Full answer text")
    context: str = Field(..., description="Source context (truncated to ~500 chars)")
    reasoning_trace: str | None = Field(None, description="Logical reasoning connections")
    tacit_inference: str | None = Field(None, description="Implicit knowledge explanation")
    validation_rationale: str | None = Field(None, description="Judge's reasoning for scores")
    generation_thinking: str | None = Field(None, description="Model thinking trace")


class TranscriptionDetail(BaseModel):
    """Full transcription detail with quality breakdown.

    Composes TranscriptionRow for summary data plus detail fields.
    """

    summary: TranscriptionRow = Field(..., description="Summary metrics and metadata")
    issues_detected: list[str] = Field(default_factory=list, description="Quality issues found")
    quality_rationale: str | None = Field(None, description="Quality assessment explanation")
    segment_count: int = Field(default=0, description="Number of transcription segments")
    total_duration_sec: float | None = Field(None, description="Total audio duration in seconds")
    transcription_text_preview: str = Field(
        default="", description="First ~500 chars of transcription text"
    )


class RunConfigResponse(BaseModel):
    """Structured configuration data for a pipeline run.

    Wraps ConfigSnapshot data with metadata about which fields are thresholds.
    """

    pipeline_id: str = Field(..., description="Pipeline run identifier")
    configs: dict[str, dict] = Field(
        default_factory=dict,
        description="Config values keyed by step name (transcription, cep)",
    )
    threshold_fields: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Fields that serve as thresholds, keyed by step name",
    )
    hardware: dict | None = Field(None, description="Hardware info from RunMetadata")
    execution: dict | None = Field(None, description="Execution environment info")


class FunnelStage(BaseModel):
    """A single stage in the data processing funnel."""

    label: str = Field(..., description="Stage display name")
    count: int = Field(..., ge=0, description="Number of items at this stage")
    drop_count: int = Field(default=0, ge=0, description="Items lost at this stage")


class FunnelData(BaseModel):
    """Data funnel showing drop-off rates at each pipeline stage."""

    pipeline_id: str = Field(..., description="Pipeline run identifier")
    stages: list[FunnelStage] = Field(default_factory=list, description="Ordered pipeline stages")


class QAFilterParams(BaseModel):
    """Query parameters for filtering QA pairs."""

    pipeline: str | None = Field(None, description="Filter by pipeline ID")
    location: str | None = Field(None, description="Filter by recording location")
    participant: str | None = Field(None, description="Filter by participant name")
    bloom_level: str | None = Field(None, description="Filter by Bloom level")
    is_valid: bool | None = Field(None, description="Filter by validity status")
    min_score: float | None = Field(None, ge=0.0, le=1.0, description="Minimum overall score")
    max_score: float | None = Field(None, ge=0.0, le=1.0, description="Maximum overall score")
    min_confidence: float | None = Field(None, ge=0.0, le=1.0, description="Minimum confidence")
    search: str | None = Field(None, description="Text search in filename/participant")
    sort_by: str = Field(default="overall_score", description="Column to sort by")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort direction")
    page: int = Field(default=1, ge=1, description="Page number")
    per_page: int = Field(default=25, ge=1, le=250, description="Items per page")


class TranscriptionFilterParams(BaseModel):
    """Query parameters for filtering transcriptions."""

    pipeline: str | None = Field(None, description="Filter by pipeline ID")
    location: str | None = Field(None, description="Filter by recording location")
    participant: str | None = Field(None, description="Filter by participant name")
    is_valid: bool | None = Field(None, description="Filter by validity status")
    min_score: float | None = Field(None, ge=0.0, le=1.0, description="Minimum overall quality")
    max_score: float | None = Field(None, ge=0.0, le=1.0, description="Maximum overall quality")
    search: str | None = Field(None, description="Text search in filename/participant")
    sort_by: str = Field(default="overall_quality", description="Column to sort by")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort direction")
    page: int = Field(default=1, ge=1, description="Page number")
    per_page: int = Field(default=25, ge=1, le=250, description="Items per page")


__all__ = [
    "FunnelData",
    "FunnelStage",
    "PaginatedResponse",
    "QAFilterParams",
    "QAPairDetail",
    "RunConfigResponse",
    "RunSummaryRow",
    "TranscriptionDetail",
    "TranscriptionFilterParams",
]
