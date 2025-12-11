"""Batch transcription with parallel processing and checkpoint support.

Provides functionality to transcribe multiple files from a catalog with
parallel processing, checkpoint/resume capability, and progress tracking.
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from gtranscriber.core.checkpoint import CheckpointManager
from gtranscriber.core.drive import DriveClient, NoAudioStreamError
from gtranscriber.core.engine import WhisperEngine
from gtranscriber.core.io import create_temp_file, save_enriched_record
from gtranscriber.core.media import get_media_duration_ms, has_audio_stream
from gtranscriber.schemas import EnrichedRecord, TranscriptionSegment

if TYPE_CHECKING:
    from gtranscriber.core.engine import TranscriptionResult

logger = logging.getLogger(__name__)

# Audio and video MIME types to process
AUDIO_VIDEO_MIME_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/flac",
    "audio/ogg",
    "audio/m4a",
    "audio/aac",
    "video/mp4",
    "video/quicktime",
    "video/mpeg",
    "video/avi",
    "video/x-msvideo",
    "video/x-matroska",
}


@dataclass
class BatchConfig:
    """Configuration for batch transcription."""

    catalog_file: Path
    output_dir: Path
    checkpoint_file: Path
    credentials_file: Path
    model_id: str = "openai/whisper-large-v3"
    num_workers: int = 1
    force_cpu: bool = False
    quantize: bool = False


@dataclass
class TranscriptionTask:
    """Task information for transcription."""

    file_id: str
    name: str
    mime_type: str
    size_bytes: int | None
    parents: list[str]
    web_content_link: str
    duration_ms: int | None


def _ensure_float(value: object, default: float) -> float:
    """Turn arbitrary values into floats with a safe fallback."""
    try:
        if value is None:
            raise TypeError
        return float(value)
    except (TypeError, ValueError):
        return default


def _create_segments_from_result(
    result: TranscriptionResult,
) -> list[TranscriptionSegment] | None:
    """Create TranscriptionSegment list from TranscriptionResult."""
    if not result.segments:
        return None

    sanitized_segments: list[TranscriptionSegment] = []
    for seg in result.segments:
        start = _ensure_float(seg.get("start"), 0.0)
        end = _ensure_float(seg.get("end"), start)

        # Ensure segment end never precedes its start
        if end < start:
            end = start

        sanitized_segments.append(
            TranscriptionSegment(
                text=seg.get("text", ""),
                start=start,
                end=end,
            )
        )

    return sanitized_segments


def transcribe_single_file(
    task: TranscriptionTask,
    config: BatchConfig,
) -> tuple[str, bool, str]:
    """Transcribe a single file (worker function for parallel processing).

    Args:
        task: TranscriptionTask with file information.
        config: BatchConfig with processing configuration.

    Returns:
        Tuple of (file_id, success, message).
    """
    try:
        logger.info(f"Processing file: {task.name} ({task.file_id})")

        # Initialize Drive client
        drive_client = DriveClient(credentials_file=str(config.credentials_file))

        # Download file with size validation
        suffix = Path(task.name).suffix
        temp_file = create_temp_file(suffix=suffix)

        try:
            drive_client.download_file(
                task.file_id,
                temp_file,
                expected_size=task.size_bytes,
                file_name=task.name,
            )
            logger.info(f"Downloaded: {task.name}")

            # Validate audio stream exists before attempting transcription
            if not has_audio_stream(temp_file):
                raise NoAudioStreamError(task.file_id, task.name, temp_file)

            # Extract duration if not provided
            duration_ms = task.duration_ms
            if duration_ms is None:
                duration_ms = get_media_duration_ms(temp_file)

            # Initialize engine and transcribe
            engine = WhisperEngine(
                model_id=config.model_id,
                force_cpu=config.force_cpu,
                quantize=config.quantize,
            )

            result = engine.transcribe(temp_file)
            logger.info(f"Transcribed: {task.name}")

            # Create enriched record
            segments = _create_segments_from_result(result)

            enriched = EnrichedRecord(
                gdrive_id=task.file_id,
                name=task.name,
                mimeType=task.mime_type,
                parents=task.parents,
                webContentLink=task.web_content_link,
                size_bytes=task.size_bytes,
                duration_milliseconds=duration_ms,
                transcription_text=result.text,
                detected_language=result.detected_language,
                language_probability=result.language_probability,
                model_id=result.model_id,
                compute_device=result.device,
                processing_duration_sec=result.processing_duration_sec,
                transcription_status="completed",
                segments=segments,
            )

            # Save result
            output_filename = f"{task.file_id}_transcription.json"
            output_path = config.output_dir / output_filename
            save_enriched_record(enriched, output_path)
            logger.info(f"Saved transcription: {output_filename}")

            return task.file_id, True, "Success"

        finally:
            # Cleanup temporary file
            if temp_file.exists():
                temp_file.unlink()

    except Exception as e:
        error_msg = f"Failed to process {task.name}: {e}"
        logger.exception(error_msg)
        return task.file_id, False, str(e)


def load_catalog(catalog_file: Path) -> list[TranscriptionTask]:
    """Load catalog CSV and filter audio/video files.

    Args:
        catalog_file: Path to catalog.csv file.

    Returns:
        List of TranscriptionTask objects.
    """
    tasks: list[TranscriptionTask] = []

    with open(catalog_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            mime_type = row.get("mime_type", "")

            # Filter only audio/video files
            if mime_type not in AUDIO_VIDEO_MIME_TYPES:
                continue

            # Parse size_bytes
            size_bytes = None
            if row.get("size_bytes"):
                try:
                    size_bytes = int(row["size_bytes"])
                except (ValueError, TypeError):
                    size_bytes = None

            # Parse duration_milliseconds
            duration_ms = None
            if row.get("duration_milliseconds"):
                try:
                    duration_ms = int(row["duration_milliseconds"])
                except (ValueError, TypeError):
                    duration_ms = None

            # Parse parents
            parents_str = row.get("parents", "[]")
            try:
                parents = json.loads(parents_str.replace("'", '"'))
            except json.JSONDecodeError:
                parents = []

            tasks.append(
                TranscriptionTask(
                    file_id=row["gdrive_id"],
                    name=row["name"],
                    mime_type=mime_type,
                    size_bytes=size_bytes,
                    parents=parents,
                    web_content_link=row.get("web_content_link", ""),
                    duration_ms=duration_ms,
                )
            )

    logger.info(f"Loaded {len(tasks)} audio/video files from catalog")
    return tasks


def run_batch_transcription(config: BatchConfig) -> None:
    """Run batch transcription with parallel processing and checkpointing.

    Args:
        config: BatchConfig with all processing parameters.
    """
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = CheckpointManager(config.checkpoint_file)

    # Load catalog
    tasks = load_catalog(config.catalog_file)

    # Filter out already completed files
    remaining_tasks = [t for t in tasks if not checkpoint.is_completed(t.file_id)]

    # Update checkpoint with total
    checkpoint.set_total_files(len(tasks))

    logger.info(f"Total files: {len(tasks)}")
    logger.info(f"Already completed: {len(tasks) - len(remaining_tasks)}")
    logger.info(f"Remaining to process: {len(remaining_tasks)}")

    if not remaining_tasks:
        logger.info("All files already transcribed!")
        return

    # Determine number of workers
    num_workers = min(config.num_workers, len(remaining_tasks))
    # Only limit workers to CPU count when using CPU mode
    if config.force_cpu and num_workers > mp.cpu_count():
        logger.warning(
            f"Requested {num_workers} workers but only {mp.cpu_count()} CPUs available"
        )
        num_workers = mp.cpu_count()
    elif num_workers > mp.cpu_count():
        logger.info(
            f"Using {num_workers} workers with GPU processing "
            f"(more than {mp.cpu_count()} CPU cores)"
        )

    logger.info(f"Using {num_workers} parallel workers")

    # Process files in parallel
    if num_workers == 1:
        # Sequential processing for single worker
        for task in remaining_tasks:
            file_id, success, message = transcribe_single_file(task, config)

            if success:
                checkpoint.mark_completed(file_id)
                logger.info(f"✓ Completed: {task.name}")
            else:
                checkpoint.mark_failed(file_id, message)
                logger.error(f"✗ Failed: {task.name} - {message}")

            completed, total = checkpoint.get_progress()
            logger.info(f"Progress: {completed}/{total} files")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(transcribe_single_file, task, config): task
                for task in remaining_tasks
            }

            # Process results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    file_id, success, message = future.result()

                    if success:
                        checkpoint.mark_completed(file_id)
                        logger.info(f"✓ Completed: {task.name}")
                    else:
                        checkpoint.mark_failed(file_id, message)
                        logger.error(f"✗ Failed: {task.name} - {message}")

                except Exception as e:
                    logger.exception(f"Task failed with exception: {task.name}")
                    checkpoint.mark_failed(task.file_id, str(e))

                completed, total = checkpoint.get_progress()
                logger.info(f"Progress: {completed}/{total} files")

    # Final summary
    completed, total = checkpoint.get_progress()
    failed_count = len(checkpoint.state.failed_files)

    logger.info("=" * 60)
    logger.info("Batch transcription completed!")
    logger.info(f"Total files: {total}")
    logger.info(f"Successfully transcribed: {completed}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success rate: {completed / total * 100:.1f}%")
    logger.info("=" * 60)

    if failed_count > 0:
        logger.warning("Failed files:")
        for file_id, error in checkpoint.state.failed_files.items():
            logger.warning(f"  - {file_id}: {error}")
