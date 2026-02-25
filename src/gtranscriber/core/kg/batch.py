"""Batch KG construction orchestrator with ResultsManager integration.

Loads transcription records, dispatches to a ``KGConstructor`` backend,
and tracks results through the versioned results system.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path  # noqa: TC003 — used at runtime for Path operations

from gtranscriber.config import KGConfig, ResultsConfig
from gtranscriber.core.checkpoint import CheckpointManager
from gtranscriber.core.kg.factory import create_kg_constructor
from gtranscriber.core.results_manager import ResultsManager
from gtranscriber.schemas import EnrichedRecord, PipelineType

logger = logging.getLogger(__name__)


def _resolve_transcription_dir(
    input_dir: Path,
    pipeline_id: str | None = None,
) -> Path:
    """Resolve the directory containing transcription JSON files.

    Args:
        input_dir: Directory provided by the caller.
        pipeline_id: Optional pipeline ID for direct resolution.

    Returns:
        The directory that contains ``*_transcription.json`` files.
    """
    if pipeline_id is not None:
        resolved = ResultsManager.resolve_outputs(
            input_dir, pipeline_id, PipelineType.TRANSCRIPTION
        )
        if resolved is not None:
            logger.info("Resolved transcription outputs for pipeline %s: %s", pipeline_id, resolved)
            return resolved

    if list(input_dir.glob("*_transcription.json")):
        return input_dir

    resolved = ResultsManager.resolve_latest_outputs(input_dir, PipelineType.TRANSCRIPTION)
    if resolved is not None:
        logger.info("Resolved transcription outputs from versioned results: %s", resolved)
        return resolved

    return input_dir


def _load_transcription_records(
    transcription_dir: Path,
) -> list[EnrichedRecord]:
    """Load and filter transcription records from JSON files.

    Records with ``is_valid=False`` are skipped.

    Args:
        transcription_dir: Directory containing ``*_transcription.json`` files.

    Returns:
        List of valid ``EnrichedRecord`` instances.
    """
    records: list[EnrichedRecord] = []
    skipped = 0

    for json_file in sorted(transcription_dir.glob("*_transcription.json")):
        try:
            data = json.loads(json_file.read_text())
            record = EnrichedRecord.model_validate(data)

            if record.is_valid is False:
                skipped += 1
                continue

            records.append(record)
        except Exception:
            logger.exception("Failed to load transcription: %s", json_file.name)
            skipped += 1

    if skipped > 0:
        logger.info("Skipped %d invalid/unreadable transcription(s)", skipped)

    return records


def run_batch_kg_construction(
    input_dir: Path,
    output_dir: Path,
    kg_config: KGConfig,
    pipeline_id: str | None = None,
) -> None:
    """Run batch knowledge graph construction with results tracking.

    Args:
        input_dir: Directory containing transcription JSON files.
        output_dir: Directory for KG outputs.
        kg_config: KG pipeline configuration.
        pipeline_id: Optional pipeline ID for versioned results layout.
    """
    results_config = ResultsConfig()
    results_mgr: ResultsManager | None = None

    # Initialize versioned results if enabled
    if results_config.enable_versioning:
        results_mgr = ResultsManager(
            results_config.base_dir,
            PipelineType.KG,
            pipeline_id=pipeline_id,
        )
        results_mgr.create_run(
            kg_config,
            input_source=str(input_dir),
            checkpoint_filename="kg_checkpoint.json",
        )
        effective_output_dir = results_mgr.outputs_dir
        effective_checkpoint_file = results_mgr.run_dir / "kg_checkpoint.json"
    else:
        effective_output_dir = output_dir
        effective_checkpoint_file = output_dir / "kg_checkpoint.json"

    effective_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize checkpoint
    checkpoint = CheckpointManager(effective_checkpoint_file)

    # Load transcription records
    transcription_dir = _resolve_transcription_dir(input_dir, pipeline_id=pipeline_id)
    records = _load_transcription_records(transcription_dir)

    if not records:
        logger.warning("No valid transcription records found in %s", transcription_dir)
        if results_mgr is not None:
            results_mgr.complete_run(success=True)
        return

    logger.info("Loaded %d transcription records for KG construction", len(records))

    # Filter out already-processed records
    remaining = [r for r in records if not checkpoint.is_completed(r.gdrive_id)]
    checkpoint.set_total_files(len(records))

    logger.info("Already completed: %d", len(records) - len(remaining))
    logger.info("Remaining to process: %d", len(remaining))

    if not remaining:
        logger.info("All records already processed!")
        if results_mgr is not None:
            results_mgr.update_progress(len(records), 0, len(records))
            results_mgr.complete_run(success=True)
        return

    # Create constructor via factory (DIP)
    constructor = create_kg_constructor(kg_config)

    error_message: str | None = None
    try:
        result = constructor.build_graph(remaining, effective_output_dir)

        # Mark all processed records as completed
        for record_id in result.source_record_ids:
            checkpoint.mark_completed(record_id)

        logger.info("=" * 60)
        logger.info("Knowledge graph construction completed!")
        logger.info("Graph file: %s", result.graph_file)
        logger.info("Nodes: %d, Edges: %d", result.node_count, result.edge_count)
        logger.info("Documents processed: %d", len(result.source_record_ids))
        logger.info("=" * 60)

    except Exception as e:
        error_message = str(e)
        logger.exception("KG construction failed")

        for record in remaining:
            checkpoint.mark_failed(record.gdrive_id, error_message)

        raise

    finally:
        completed, total = checkpoint.get_progress()
        failed_count = len(checkpoint.state.failed_files)

        if results_mgr is not None:
            results_mgr.update_progress(completed, failed_count, total)
            success = error_message is None and failed_count < total
            results_mgr.complete_run(success=success, error=error_message)
