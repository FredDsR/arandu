"""Business logic for the report dashboard API.

Encapsulates filtering, pagination, funnel computation, and data
transformation. Receives ResultsCollector via dependency injection
to keep data access concerns separate.
"""

from __future__ import annotations

import csv
import logging
from io import StringIO
from typing import TYPE_CHECKING

from .api_schemas import (
    FunnelData,
    FunnelStage,
    PaginatedResponse,
    QAFilterParams,
    QAPairDetail,
    RunConfigResponse,
    TranscriptionDetail,
    TranscriptionFilterParams,
)
from .dataset import QAPairRow, RunSummaryRow, TranscriptionRow, build_dataset

if TYPE_CHECKING:
    from .collector import ResultsCollector

logger = logging.getLogger(__name__)

_THRESHOLD_FIELDS: list[str] = ["validation_threshold", "quality_threshold"]
_TEXT_PREVIEW_CHARS: int = 500


class ReportService:
    """Service layer for report data access and transformation.

    Args:
        collector: ResultsCollector for filesystem data access.
    """

    def __init__(self, collector: ResultsCollector) -> None:
        """Initialise with a ResultsCollector.

        Args:
            collector: ResultsCollector for filesystem data access.
        """
        self._collector = collector
        self._dataset = None

    def _get_dataset(self) -> object:
        """Return (and lazily build) the cached ReportDataset.

        Returns:
            ReportDataset built from all runs.
        """
        if self._dataset is None:
            reports = self._collector.load_all_runs()
            self._dataset = build_dataset(reports)
        return self._dataset

    def list_runs(self) -> list[RunSummaryRow]:
        """Return all discovered run summaries.

        Returns:
            List of RunSummaryRow objects for all pipeline runs.
        """
        return self._get_dataset().runs  # type: ignore[union-attr]

    def get_run_summary(self, pipeline_id: str) -> RunSummaryRow:
        """Return summary for a single run.

        Args:
            pipeline_id: The pipeline ID to look up.

        Returns:
            RunSummaryRow for the requested pipeline.

        Raises:
            KeyError: If the pipeline_id is not found.
        """
        for row in self.list_runs():
            if row.pipeline_id == pipeline_id:
                return row
        raise KeyError(f"Pipeline not found: {pipeline_id}")

    def get_run_config(self, pipeline_id: str) -> RunConfigResponse:
        """Return configuration data for a pipeline run.

        Calls collector.load_all_run_configs to load configs for all steps.
        Identifies threshold fields by checking against a known list.

        Args:
            pipeline_id: The pipeline ID to load configs for.

        Returns:
            RunConfigResponse with step configs and threshold metadata.
        """
        configs_raw = self._collector.load_all_run_configs(pipeline_id)

        configs: dict[str, dict] = {}
        threshold_fields: dict[str, list[str]] = {}

        for step, snapshot in configs_raw.items():
            values = snapshot.config_values
            configs[step] = values
            found = [f for f in _THRESHOLD_FIELDS if f in values]
            if found:
                threshold_fields[step] = found

        # Try to fetch hardware / execution from run metadata
        hardware: dict | None = None
        execution: dict | None = None
        try:
            report = self._collector.load_run(pipeline_id)
            meta = report.cep_metadata or report.transcription_metadata
            if meta:
                if hasattr(meta, "hardware") and meta.hardware:
                    hardware = meta.hardware.model_dump()
                if hasattr(meta, "execution") and meta.execution:
                    execution = meta.execution.model_dump()
        except Exception:
            logger.debug("Could not load run metadata for config response: %s", pipeline_id)

        return RunConfigResponse(
            pipeline_id=pipeline_id,
            configs=configs,
            threshold_fields=threshold_fields,
            hardware=hardware,
            execution=execution,
        )

    def list_qa_pairs(self, filters: QAFilterParams) -> PaginatedResponse[QAPairRow]:
        """Return a paginated, filtered list of QA pairs.

        Args:
            filters: QAFilterParams specifying filters, sort, and pagination.

        Returns:
            PaginatedResponse containing the matching QA pairs.
        """
        rows: list[QAPairRow] = list(self._get_dataset().qa_pairs)  # type: ignore[union-attr]

        # Apply filters
        if filters.pipeline is not None:
            rows = [r for r in rows if r.pipeline_id == filters.pipeline]
        if filters.location is not None:
            rows = [r for r in rows if r.location == filters.location]
        if filters.participant is not None:
            rows = [r for r in rows if r.participant_name == filters.participant]
        if filters.bloom_level is not None:
            rows = [r for r in rows if r.bloom_level == filters.bloom_level]
        if filters.is_valid is not None:
            rows = [r for r in rows if r.is_valid == filters.is_valid]
        if filters.min_score is not None:
            rows = [
                r
                for r in rows
                if r.overall_score is not None and r.overall_score >= filters.min_score
            ]
        if filters.max_score is not None:
            rows = [
                r
                for r in rows
                if r.overall_score is not None and r.overall_score <= filters.max_score
            ]
        if filters.min_confidence is not None:
            rows = [
                r
                for r in rows
                if r.confidence is not None and r.confidence >= filters.min_confidence
            ]
        if filters.search is not None:
            term = filters.search.lower()
            rows = [
                r
                for r in rows
                if term in r.source_filename.lower()
                or (r.participant_name and term in r.participant_name.lower())
            ]

        # Sort
        reverse = filters.sort_order == "desc"
        sort_attr = filters.sort_by

        def _qa_sort_key(row: QAPairRow) -> tuple:
            val = getattr(row, sort_attr, None)
            return (val is None, val if val is not None else 0)

        rows = sorted(rows, key=_qa_sort_key, reverse=reverse)

        return _paginate(rows, filters.page, filters.per_page)

    def get_qa_detail(self, pipeline_id: str, source_filename: str, index: int) -> QAPairDetail:
        """Return full detail for a single QA pair.

        Args:
            pipeline_id: The pipeline ID.
            source_filename: Source filename the QA pair belongs to.
            index: Zero-based index of the QA pair within the record.

        Returns:
            QAPairDetail composed from QAPairRow summary and text fields.

        Raises:
            KeyError: If the record or index is not found.
        """
        record = self._collector.load_qa_record(pipeline_id, source_filename)
        if record is None:
            raise KeyError(f"QA record not found: {pipeline_id}/{source_filename}")

        if index < 0 or index >= len(record.qa_pairs):
            raise KeyError(
                f"QA pair index {index} out of range for {pipeline_id}/{source_filename}"
            )

        qa_pair = record.qa_pairs[index]
        validation = getattr(qa_pair, "validation", None)

        # Build the summary row (reuse build logic)
        from .collector import RunReport
        from .dataset import _build_qa_rows

        # Minimal RunReport to reuse the helper
        temp_report = RunReport(pipeline_id=pipeline_id, cep_records=[record])
        qa_rows: list[QAPairRow] = []
        _build_qa_rows(temp_report, qa_rows)
        if index >= len(qa_rows):
            raise KeyError(f"QA pair index {index} out of range after building rows")
        summary = qa_rows[index]

        return QAPairDetail(
            summary=summary,
            question=getattr(qa_pair, "question", ""),
            answer=getattr(qa_pair, "answer", ""),
            context=(record.transcription_text or "")[:_TEXT_PREVIEW_CHARS],
            reasoning_trace=getattr(qa_pair, "reasoning_trace", None),
            tacit_inference=getattr(qa_pair, "tacit_inference", None),
            validation_rationale=validation.judge_rationale if validation else None,
            generation_thinking=getattr(qa_pair, "generation_thinking", None),
        )

    def list_transcriptions(
        self, filters: TranscriptionFilterParams
    ) -> PaginatedResponse[TranscriptionRow]:
        """Return a paginated, filtered list of transcriptions.

        Args:
            filters: TranscriptionFilterParams specifying filters, sort, and pagination.

        Returns:
            PaginatedResponse containing the matching transcription rows.
        """
        rows: list[TranscriptionRow] = list(self._get_dataset().transcriptions)  # type: ignore[union-attr]

        # Apply filters
        if filters.pipeline is not None:
            rows = [r for r in rows if r.pipeline_id == filters.pipeline]
        if filters.location is not None:
            rows = [r for r in rows if r.location == filters.location]
        if filters.participant is not None:
            rows = [r for r in rows if r.participant_name == filters.participant]
        if filters.is_valid is not None:
            rows = [r for r in rows if r.is_valid == filters.is_valid]
        if filters.min_score is not None:
            rows = [
                r
                for r in rows
                if r.overall_quality is not None and r.overall_quality >= filters.min_score
            ]
        if filters.max_score is not None:
            rows = [
                r
                for r in rows
                if r.overall_quality is not None and r.overall_quality <= filters.max_score
            ]
        if filters.search is not None:
            term = filters.search.lower()
            rows = [
                r
                for r in rows
                if term in r.source_filename.lower()
                or (r.participant_name and term in r.participant_name.lower())
            ]

        # Sort
        reverse = filters.sort_order == "desc"
        sort_attr = filters.sort_by

        def _trans_sort_key(row: TranscriptionRow) -> tuple:
            val = getattr(row, sort_attr, None)
            return (val is None, val if val is not None else 0)

        rows = sorted(rows, key=_trans_sort_key, reverse=reverse)

        return _paginate(rows, filters.page, filters.per_page)

    def get_transcription_detail(
        self, pipeline_id: str, source_filename: str
    ) -> TranscriptionDetail:
        """Return full detail for a single transcription.

        Args:
            pipeline_id: The pipeline ID.
            source_filename: Source filename of the transcription.

        Returns:
            TranscriptionDetail composed from TranscriptionRow and record fields.

        Raises:
            KeyError: If the transcription record is not found.
        """
        record = self._collector.load_transcription_record(pipeline_id, source_filename)
        if record is None:
            raise KeyError(f"Transcription record not found: {pipeline_id}/{source_filename}")

        # Find matching row in dataset for summary
        summary: TranscriptionRow | None = None
        for row in self._get_dataset().transcriptions:  # type: ignore[union-attr]
            if row.pipeline_id == pipeline_id and row.source_filename == source_filename:
                summary = row
                break

        if summary is None:
            # Build from record directly
            from .collector import RunReport
            from .dataset import _build_transcription_rows

            temp_report = RunReport(pipeline_id=pipeline_id, transcription_records=[record])
            rows: list[TranscriptionRow] = []
            _build_transcription_rows(temp_report, rows)
            if not rows:
                raise KeyError(f"Could not build summary for {pipeline_id}/{source_filename}")
            summary = rows[0]

        quality = record.transcription_quality
        issues: list[str] = []
        rationale: str | None = None
        if quality:
            issues = quality.issues_detected if hasattr(quality, "issues_detected") else []
            rationale = quality.quality_rationale if hasattr(quality, "quality_rationale") else None

        segments = record.segments or []
        total_duration: float | None = None
        if segments:
            last_seg = max(segments, key=lambda s: s.end)
            total_duration = last_seg.end

        preview = (record.transcription_text or "")[:_TEXT_PREVIEW_CHARS]

        return TranscriptionDetail(
            summary=summary,
            issues_detected=issues,
            quality_rationale=rationale,
            segment_count=len(segments),
            total_duration_sec=total_duration,
            transcription_text_preview=preview,
        )

    def get_funnel(self, pipeline_id: str) -> FunnelData:
        """Return the data funnel for a pipeline run.

        Args:
            pipeline_id: The pipeline ID to compute the funnel for.

        Returns:
            FunnelData with ordered stages and drop-off counts.

        Raises:
            KeyError: If the pipeline_id is not found.
        """
        run_summary = self.get_run_summary(pipeline_id)

        total_transcriptions = run_summary.valid_transcriptions + run_summary.invalid_transcriptions
        total_qa = run_summary.valid_qa_pairs + run_summary.invalid_qa_pairs

        stages: list[FunnelStage] = [
            FunnelStage(
                label="Total Transcriptions",
                count=total_transcriptions,
                drop_count=0,
            ),
            FunnelStage(
                label="Valid Transcriptions",
                count=run_summary.valid_transcriptions,
                drop_count=run_summary.invalid_transcriptions,
            ),
            FunnelStage(
                label="Total QA Pairs",
                count=total_qa,
                drop_count=max(0, run_summary.valid_transcriptions - total_qa),
            ),
            FunnelStage(
                label="Valid QA Pairs",
                count=run_summary.valid_qa_pairs,
                drop_count=run_summary.invalid_qa_pairs,
            ),
        ]

        return FunnelData(pipeline_id=pipeline_id, stages=stages)

    def export_csv(self, data_type: str, filters: dict) -> str:
        """Generate CSV content for the requested data type with optional filters.

        Args:
            data_type: Either ``"qa"`` or ``"transcriptions"``.
            filters: Dict of filter parameters (passed through to filter logic).

        Returns:
            CSV content as a string.

        Raises:
            ValueError: If data_type is unsupported.
        """
        output = StringIO()

        if data_type == "qa":
            qa_filters = QAFilterParams(**{k: v for k, v in filters.items() if v is not None})
            qa_filters.per_page = 100
            qa_filters.page = 1
            # Collect all pages
            all_rows: list[QAPairRow] = []
            while True:
                page = self.list_qa_pairs(qa_filters)
                all_rows.extend(page.items)
                if qa_filters.page >= page.total_pages:
                    break
                qa_filters.page += 1
            if all_rows:
                writer = csv.DictWriter(output, fieldnames=list(QAPairRow.model_fields))
                writer.writeheader()
                for row in all_rows:
                    writer.writerow(row.model_dump())
        elif data_type == "transcriptions":
            trans_filters = TranscriptionFilterParams(
                **{k: v for k, v in filters.items() if v is not None}
            )
            trans_filters.per_page = 100
            trans_filters.page = 1
            all_trans_rows: list[TranscriptionRow] = []
            while True:
                page_t = self.list_transcriptions(trans_filters)
                all_trans_rows.extend(page_t.items)
                if trans_filters.page >= page_t.total_pages:
                    break
                trans_filters.page += 1
            if all_trans_rows:
                writer = csv.DictWriter(output, fieldnames=list(TranscriptionRow.model_fields))
                writer.writeheader()
                for row in all_trans_rows:
                    writer.writerow(row.model_dump())
        else:
            raise ValueError(f"Unsupported data_type: {data_type!r}. Use 'qa' or 'transcriptions'")

        return output.getvalue()


def _paginate(rows: list, page: int, per_page: int) -> PaginatedResponse:
    """Slice a list into a single page and wrap in PaginatedResponse.

    Args:
        rows: Full (already filtered and sorted) list of items.
        page: 1-indexed page number.
        per_page: Number of items per page.

    Returns:
        PaginatedResponse for the requested page.
    """
    total = len(rows)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    end = start + per_page
    return PaginatedResponse(
        items=rows[start:end],
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
    )
