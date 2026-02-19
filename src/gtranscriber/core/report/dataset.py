"""Flat data models and dataset builder for report generation.

Flattens the nested RunReport structure into tabular rows suitable for
JSON serialization and client-side JavaScript consumption.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, computed_field

if TYPE_CHECKING:
    from .collector import RunReport


class QAPairRow(BaseModel):
    """Flat row representing a single QA pair with all associated metadata."""

    pipeline_id: str
    source_filename: str
    participant_name: str | None = None
    location: str | None = None
    recording_date: str | None = None
    bloom_level: str | None = None
    is_multi_hop: bool = False
    hop_count: int | None = None
    confidence: float | None = None
    faithfulness: float | None = None
    bloom_calibration: float | None = None
    informativeness: float | None = None
    self_containedness: float | None = None
    overall_score: float | None = None
    model_id: str | None = None
    validator_model_id: str | None = None
    provider: str | None = None
    is_valid: bool = True


class TranscriptionRow(BaseModel):
    """Flat row representing a single transcription record."""

    pipeline_id: str
    source_filename: str
    participant_name: str | None = None
    location: str | None = None
    recording_date: str | None = None
    is_valid: bool | None = None
    overall_quality: float | None = None
    script_match: float | None = None
    repetition: float | None = None
    segment_quality: float | None = None
    content_density: float | None = None
    processing_duration_sec: float | None = None
    model_id: str | None = None
    detected_language: str | None = None


class RunSummaryRow(BaseModel):
    """Flat row representing a single pipeline run summary."""

    pipeline_id: str
    steps_run: list[str] = Field(default_factory=list)
    status: str = "unknown"
    duration_seconds: float | None = None
    success_rate: float | None = None
    completed_items: int = 0
    total_items: int = 0
    created_at: str | None = None
    device_type: str | None = None
    gpu_name: str | None = None
    is_slurm: bool = False


class ReportDataset(BaseModel):
    """Container for all flattened report data with derived filter lists."""

    qa_pairs: list[QAPairRow] = Field(default_factory=list)
    transcriptions: list[TranscriptionRow] = Field(default_factory=list)
    runs: list[RunSummaryRow] = Field(default_factory=list)
    generated_at: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pipeline_ids(self) -> list[str]:
        """Unique pipeline IDs across all data."""
        return sorted({r.pipeline_id for r in self.runs})

    @computed_field  # type: ignore[prop-decorator]
    @property
    def locations(self) -> list[str]:
        """Unique locations across transcriptions and QA pairs."""
        locs: set[str] = set()
        for t in self.transcriptions:
            if t.location:
                locs.add(t.location)
        for q in self.qa_pairs:
            if q.location:
                locs.add(q.location)
        return sorted(locs)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def participants(self) -> list[str]:
        """Unique participant names across transcriptions and QA pairs."""
        names: set[str] = set()
        for t in self.transcriptions:
            if t.participant_name:
                names.add(t.participant_name)
        for q in self.qa_pairs:
            if q.participant_name:
                names.add(q.participant_name)
        return sorted(names)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bloom_levels(self) -> list[str]:
        """Unique Bloom taxonomy levels across QA pairs."""
        return sorted({q.bloom_level for q in self.qa_pairs if q.bloom_level})


def build_dataset(reports: list[RunReport]) -> ReportDataset:
    """Build a flat ReportDataset from nested RunReport objects.

    Iterates over RunReport objects and flattens nested Pydantic models
    into tabular rows. Excludes large text fields to keep JSON small.

    Args:
        reports: List of RunReport objects to flatten.

    Returns:
        ReportDataset containing all flattened data.
    """
    qa_rows: list[QAPairRow] = []
    transcription_rows: list[TranscriptionRow] = []
    run_rows: list[RunSummaryRow] = []

    for report in reports:
        _build_run_summary(report, run_rows)
        _build_transcription_rows(report, transcription_rows)
        _build_qa_rows(report, qa_rows)

    return ReportDataset(
        qa_pairs=qa_rows,
        transcriptions=transcription_rows,
        runs=run_rows,
        generated_at=datetime.now(tz=UTC).isoformat(),
    )


def _build_run_summary(report: RunReport, run_rows: list[RunSummaryRow]) -> None:
    """Extract run summary from a RunReport.

    Args:
        report: Source RunReport.
        run_rows: Target list to append the row to.
    """
    metadata = report.cep_metadata or report.transcription_metadata

    row = RunSummaryRow(
        pipeline_id=report.pipeline_id,
        steps_run=report.pipeline.steps_run if report.pipeline else [],
        status=metadata.status.value if metadata else "unknown",
        duration_seconds=metadata.duration_seconds if metadata else None,
        success_rate=metadata.success_rate if metadata else None,
        completed_items=metadata.completed_items if metadata else 0,
        total_items=metadata.total_items if metadata else 0,
    )

    if report.pipeline and hasattr(report.pipeline, "created_at"):
        row.created_at = report.pipeline.created_at.isoformat()

    if metadata and hasattr(metadata, "hardware"):
        row.device_type = metadata.hardware.device_type
        row.gpu_name = metadata.hardware.gpu_name

    if metadata and hasattr(metadata, "execution"):
        row.is_slurm = metadata.execution.is_slurm

    run_rows.append(row)


def _build_transcription_rows(
    report: RunReport, transcription_rows: list[TranscriptionRow]
) -> None:
    """Extract transcription rows from a RunReport.

    Args:
        report: Source RunReport.
        transcription_rows: Target list to append rows to.
    """
    for record in report.transcription_records:
        source = record.source_metadata
        quality = record.transcription_quality

        row = TranscriptionRow(
            pipeline_id=report.pipeline_id,
            source_filename=record.name,
            participant_name=source.participant_name if source else None,
            location=source.location if source else None,
            recording_date=source.recording_date if source else None,
            is_valid=record.is_valid,
            overall_quality=quality.overall_score if quality else None,
            script_match=quality.script_match_score if quality else None,
            repetition=quality.repetition_score if quality else None,
            segment_quality=quality.segment_quality_score if quality else None,
            content_density=quality.content_density_score if quality else None,
            processing_duration_sec=record.processing_duration_sec,
            model_id=record.model_id,
            detected_language=record.detected_language,
        )
        transcription_rows.append(row)


def _build_qa_rows(report: RunReport, qa_rows: list[QAPairRow]) -> None:
    """Extract QA pair rows from a RunReport.

    Args:
        report: Source RunReport.
        qa_rows: Target list to append rows to.
    """
    for cep_record in report.cep_records:
        source = cep_record.source_metadata

        for qa_pair in cep_record.qa_pairs:
            validation = getattr(qa_pair, "validation", None)

            row = QAPairRow(
                pipeline_id=report.pipeline_id,
                source_filename=cep_record.source_filename,
                participant_name=source.participant_name if source else None,
                location=source.location if source else None,
                recording_date=source.recording_date if source else None,
                bloom_level=qa_pair.bloom_level if hasattr(qa_pair, "bloom_level") else None,
                is_multi_hop=getattr(qa_pair, "is_multi_hop", False),
                hop_count=getattr(qa_pair, "hop_count", None),
                confidence=getattr(qa_pair, "confidence", None),
                faithfulness=validation.faithfulness if validation else None,
                bloom_calibration=validation.bloom_calibration if validation else None,
                informativeness=validation.informativeness if validation else None,
                self_containedness=validation.self_containedness if validation else None,
                overall_score=validation.overall_score if validation else None,
                model_id=cep_record.model_id,
                validator_model_id=cep_record.validator_model_id,
                provider=cep_record.provider,
                is_valid=getattr(qa_pair, "is_valid", True),
            )
            qa_rows.append(row)
