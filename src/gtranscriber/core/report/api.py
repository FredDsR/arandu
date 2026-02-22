"""FastAPI route handlers for the report dashboard API.

Thin controllers that parse HTTP parameters and delegate to
ReportService. No business logic lives here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from .service import ReportService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


def create_app(results_dir: Path) -> FastAPI:
    """Create the FastAPI application for the report dashboard.

    Args:
        results_dir: Path to results directory containing pipeline runs.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="G-Transcriber Report Dashboard",
        description="Interactive dashboard for pipeline results exploration.",
        version="1.0.0",
    )

    app.state.results_dir = results_dir
    app.include_router(router)

    return app


def get_report_service() -> ReportService:
    """Provide ReportService with injected ResultsCollector.

    Returns:
        Configured ReportService instance.

    Note:
        Intended to be overridden by ``app.dependency_overrides`` so that
        the ``results_dir`` stored in ``app.state`` is used.
    """
    raise RuntimeError(  # pragma: no cover
        "get_report_service must be overridden via app.dependency_overrides"
    )


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------


@router.get("/runs", response_model=list)
def list_runs(
    service: ReportService = Depends(get_report_service),
) -> list:
    """Return all discovered pipeline run summaries.

    Args:
        service: Injected ReportService.

    Returns:
        List of RunSummaryRow dicts.
    """
    try:
        return [r.model_dump() for r in service.list_runs()]
    except Exception:
        logger.exception("Failed to list runs")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/runs/{pipeline_id}", response_model=dict)
def get_run_detail(
    pipeline_id: str,
    service: ReportService = Depends(get_report_service),
) -> dict:
    """Return summary for a single pipeline run.

    Args:
        pipeline_id: Pipeline run identifier.
        service: Injected ReportService.

    Returns:
        RunSummaryRow as a dict.
    """
    try:
        return service.get_run_summary(pipeline_id).model_dump()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}") from None
    except Exception:
        logger.exception("Failed to get run detail for %s", pipeline_id)
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/runs/{pipeline_id}/config", response_model=dict)
def get_run_config(
    pipeline_id: str,
    service: ReportService = Depends(get_report_service),
) -> dict:
    """Return configuration data for a pipeline run.

    Args:
        pipeline_id: Pipeline run identifier.
        service: Injected ReportService.

    Returns:
        RunConfigResponse as a dict.
    """
    try:
        return service.get_run_config(pipeline_id).model_dump()
    except Exception:
        logger.exception("Failed to get run config for %s", pipeline_id)
        raise HTTPException(status_code=500, detail="Internal server error") from None


# ---------------------------------------------------------------------------
# QA pair endpoints
# ---------------------------------------------------------------------------


@router.get("/qa", response_model=dict)
def list_qa_pairs(
    pipeline: str | None = Query(default=None),
    location: str | None = Query(default=None),
    participant: str | None = Query(default=None),
    bloom_level: str | None = Query(default=None),
    is_valid: bool | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=0.0, le=1.0),
    max_score: float | None = Query(default=None, ge=0.0, le=1.0),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    search: str | None = Query(default=None),
    sort_by: str = Query(default="overall_score"),
    sort_order: str = Query(default="desc", pattern="^(asc|desc)$"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=25, ge=1, le=100),
    service: ReportService = Depends(get_report_service),
) -> dict:
    """Return a paginated, filtered list of QA pairs.

    Args:
        pipeline: Filter by pipeline ID.
        location: Filter by recording location.
        participant: Filter by participant name.
        bloom_level: Filter by Bloom taxonomy level.
        is_valid: Filter by validity status.
        min_score: Minimum overall score.
        max_score: Maximum overall score.
        min_confidence: Minimum confidence score.
        search: Text search in filename/participant.
        sort_by: Column to sort by.
        sort_order: Sort direction (asc/desc).
        page: Page number.
        per_page: Items per page.
        service: Injected ReportService.

    Returns:
        PaginatedResponse[QAPairRow] as a dict.
    """
    from .api_schemas import QAFilterParams

    try:
        filters = QAFilterParams(
            pipeline=pipeline,
            location=location,
            participant=participant,
            bloom_level=bloom_level,
            is_valid=is_valid,
            min_score=min_score,
            max_score=max_score,
            min_confidence=min_confidence,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            per_page=per_page,
        )
        result = service.list_qa_pairs(filters)
        return result.model_dump()
    except Exception:
        logger.exception("Failed to list QA pairs")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/qa/{pipeline_id}/{source_filename}/{index}", response_model=dict)
def get_qa_detail(
    pipeline_id: str,
    source_filename: str,
    index: int,
    service: ReportService = Depends(get_report_service),
) -> dict:
    """Return full detail for a single QA pair.

    Args:
        pipeline_id: Pipeline run identifier.
        source_filename: Source filename the QA pair belongs to.
        index: Zero-based index of the QA pair within the record.
        service: Injected ReportService.

    Returns:
        QAPairDetail as a dict.
    """
    try:
        return service.get_qa_detail(pipeline_id, source_filename, index).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        logger.exception(
            "Failed to get QA detail for %s/%s/%d", pipeline_id, source_filename, index
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


# ---------------------------------------------------------------------------
# Transcription endpoints
# ---------------------------------------------------------------------------


@router.get("/transcriptions", response_model=dict)
def list_transcriptions(
    pipeline: str | None = Query(default=None),
    location: str | None = Query(default=None),
    participant: str | None = Query(default=None),
    is_valid: bool | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=0.0, le=1.0),
    max_score: float | None = Query(default=None, ge=0.0, le=1.0),
    search: str | None = Query(default=None),
    sort_by: str = Query(default="overall_quality"),
    sort_order: str = Query(default="desc", pattern="^(asc|desc)$"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=25, ge=1, le=100),
    service: ReportService = Depends(get_report_service),
) -> dict:
    """Return a paginated, filtered list of transcriptions.

    Args:
        pipeline: Filter by pipeline ID.
        location: Filter by recording location.
        participant: Filter by participant name.
        is_valid: Filter by validity status.
        min_score: Minimum overall quality score.
        max_score: Maximum overall quality score.
        search: Text search in filename/participant.
        sort_by: Column to sort by.
        sort_order: Sort direction (asc/desc).
        page: Page number.
        per_page: Items per page.
        service: Injected ReportService.

    Returns:
        PaginatedResponse[TranscriptionRow] as a dict.
    """
    from .api_schemas import TranscriptionFilterParams

    try:
        filters = TranscriptionFilterParams(
            pipeline=pipeline,
            location=location,
            participant=participant,
            is_valid=is_valid,
            min_score=min_score,
            max_score=max_score,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            per_page=per_page,
        )
        result = service.list_transcriptions(filters)
        return result.model_dump()
    except Exception:
        logger.exception("Failed to list transcriptions")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/transcriptions/{pipeline_id}/{source_filename}", response_model=dict)
def get_transcription_detail(
    pipeline_id: str,
    source_filename: str,
    service: ReportService = Depends(get_report_service),
) -> dict:
    """Return full detail for a single transcription.

    Args:
        pipeline_id: Pipeline run identifier.
        source_filename: Source filename of the transcription.
        service: Injected ReportService.

    Returns:
        TranscriptionDetail as a dict.
    """
    try:
        return service.get_transcription_detail(pipeline_id, source_filename).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        logger.exception(
            "Failed to get transcription detail for %s/%s", pipeline_id, source_filename
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


# ---------------------------------------------------------------------------
# Funnel endpoint
# ---------------------------------------------------------------------------


@router.get("/funnel/{pipeline_id}", response_model=dict)
def get_funnel(
    pipeline_id: str,
    service: ReportService = Depends(get_report_service),
) -> dict:
    """Return funnel data for a pipeline run.

    Args:
        pipeline_id: Pipeline run identifier.
        service: Injected ReportService.

    Returns:
        FunnelData as a dict.
    """
    try:
        return service.get_funnel(pipeline_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        logger.exception("Failed to get funnel for %s", pipeline_id)
        raise HTTPException(status_code=500, detail="Internal server error") from None


# ---------------------------------------------------------------------------
# Export endpoint
# ---------------------------------------------------------------------------


@router.get("/export/{data_type}")
def export_csv(
    data_type: str,
    pipeline: str | None = Query(default=None),
    is_valid: bool | None = Query(default=None),
    service: ReportService = Depends(get_report_service),
) -> StreamingResponse:
    """Export filtered data as CSV.

    Args:
        data_type: Data type to export (``"qa"`` or ``"transcriptions"``).
        pipeline: Optional pipeline filter.
        is_valid: Optional validity filter.
        service: Injected ReportService.

    Returns:
        StreamingResponse with CSV content.
    """
    try:
        filters: dict = {}
        if pipeline is not None:
            filters["pipeline"] = pipeline
        if is_valid is not None:
            filters["is_valid"] = is_valid
        csv_content = service.export_csv(data_type, filters)
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={data_type}.csv"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Failed to export CSV for %s", data_type)
        raise HTTPException(status_code=500, detail="Internal server error") from None
