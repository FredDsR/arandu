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

from gtranscriber.config import ResultsConfig, TranscriberConfig
from gtranscriber.core.checkpoint import CheckpointManager
from gtranscriber.core.drive import DriveClient, NoAudioStreamError
from gtranscriber.core.engine import WhisperEngine
from gtranscriber.core.io import create_temp_file, save_enriched_record
from gtranscriber.core.media import (
    AudioExtractionError,
    CorruptedMediaError,
    extract_audio,
    get_media_duration_ms,
    has_audio_stream,
    requires_audio_extraction,
)
from gtranscriber.core.transcription_validator import validate_enriched_record
from gtranscriber.core.results_manager import ResultsManager
from gtranscriber.schemas import EnrichedRecord, PipelineType, TranscriptionSegment

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

# Global engine instance per worker process
_worker_engine: WhisperEngine | None = None


def _parse_parents_from_string(parents_str: str | list[str]) -> list[str]:
    """Parse parents field from string or list format (shared utility).

    Uses the same logic as InputRecord.parse_parents for consistency.

    Args:
        parents_str: String or list representation of parents.

    Returns:
        List of parent folder IDs.
    """
    if isinstance(parents_str, str):
        try:
            # Handle single-quoted JSON strings
            result = json.loads(parents_str.replace("'", '"'))
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []
    if isinstance(parents_str, list):
        return parents_str
    return []


def _init_worker(model_id: str, force_cpu: bool, quantize: bool, language: str | None) -> None:
    """Initialize worker process with a WhisperEngine instance.

    This function is called once per worker process to load the model,
    avoiding the overhead of loading the model for each file.

    Args:
        model_id: Hugging Face model ID for transcription.
        force_cpu: Force CPU execution.
        quantize: Enable 8-bit quantization.
        language: Language code for transcription (e.g., 'pt' for Portuguese).
    """
    global _worker_engine
    _worker_engine = WhisperEngine(
        model_id=model_id,
        force_cpu=force_cpu,
        quantize=quantize,
        language=language,
    )
    logger.info(f"Worker initialized with model {model_id}")


@dataclass
class BatchConfig:
    """Configuration for batch transcription."""

    catalog_file: Path
    output_dir: Path
    checkpoint_file: Path
    credentials_file: Path
    token_file: Path
    model_id: str = "openai/whisper-large-v3"
    num_workers: int = 1
    force_cpu: bool = False
    quantize: bool = False
    language: str | None = None

    @classmethod
    def from_transcriber_config(
        cls,
        catalog_file: Path,
        output_dir: Path,
        checkpoint_file: Path,
        config: TranscriberConfig | None = None,
        num_workers: int = 1,
    ) -> BatchConfig:
        """Create BatchConfig from TranscriberConfig.

        Args:
            catalog_file: Path to catalog CSV file
            output_dir: Output directory for results
            checkpoint_file: Path to checkpoint file
            config: TranscriberConfig instance. If not provided, will be loaded from environment.
            num_workers: Number of parallel workers

        Returns:
            BatchConfig instance
        """
        if config is None:
            config = TranscriberConfig()

        return cls(
            catalog_file=catalog_file,
            output_dir=output_dir,
            checkpoint_file=checkpoint_file,
            credentials_file=Path(config.credentials),
            token_file=Path(config.token),
            model_id=config.model_id,
            num_workers=num_workers,
            force_cpu=config.force_cpu,
            quantize=config.quantize,
            language=config.language,
        )


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
        return float(value)  # ty:ignore[invalid-argument-type]
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

    Uses the global _worker_engine that was initialized once per worker process.
    For video files (MP4, MOV, etc.), audio is extracted to WAV format before
    transcription to ensure compatibility with the Whisper audio backend.

    Args:
        task: TranscriptionTask with file information.
        config: BatchConfig with processing configuration.

    Returns:
        Tuple of (file_id, success, message).
    """
    global _worker_engine

    try:
        logger.info(f"Processing file: {task.name} ({task.file_id})")

        # Initialize Drive client
        drive_client = DriveClient(
            credentials_file=str(config.credentials_file),
            token_file=str(config.token_file),
        )

        # Download file with size validation
        suffix = Path(task.name).suffix
        temp_file = create_temp_file(suffix=suffix)
        extracted_audio_file: Path | None = None

        try:
            drive_client.download_file(
                task.file_id,
                temp_file,
                expected_size=task.size_bytes,
                file_name=task.name,
            )
            logger.info(f"Downloaded: {task.name}")

            # Determine which file to use for transcription
            transcription_file = temp_file

            # For video files, extract audio to WAV format for compatibility
            if requires_audio_extraction(task.mime_type):
                logger.info(f"Extracting audio from video: {task.name}")
                extracted_audio_file = create_temp_file(suffix=".wav")
                try:
                    extract_audio(temp_file, extracted_audio_file)
                    transcription_file = extracted_audio_file
                    logger.info(f"Audio extracted successfully: {task.name}")
                except CorruptedMediaError:
                    # Re-raise with original context
                    raise
                except AudioExtractionError:
                    # Re-raise with original context
                    raise
            else:
                # For audio files, validate audio stream exists
                if not has_audio_stream(temp_file):
                    raise NoAudioStreamError(task.file_id, task.name, temp_file)

            # Extract duration if not provided
            duration_ms = task.duration_ms
            if duration_ms is None:
                duration_ms = get_media_duration_ms(temp_file)

            # Use the pre-initialized engine for this worker process
            # For sequential processing (single worker), initialize on first use
            if _worker_engine is None:
                _worker_engine = WhisperEngine(
                    model_id=config.model_id,
                    force_cpu=config.force_cpu,
                    quantize=config.quantize,
                    language=config.language,
                )

            result = _worker_engine.transcribe(transcription_file)
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

            # Quality validation (lightweight, CPU-only)
            validate_enriched_record(enriched)

            if enriched.is_valid is False:
                logger.warning(
                    f"Quality check failed for {task.name}: "
                    f"{enriched.transcription_quality.issues_detected}"
                )

            # Save result
            output_filename = f"{task.file_id}_transcription.json"
            output_path = config.output_dir / output_filename
            save_enriched_record(enriched, output_path)
            logger.info(f"Saved transcription: {output_filename}")

            return task.file_id, True, "Success"

        finally:
            # Cleanup temporary files
            if temp_file.exists():
                temp_file.unlink()
            if extracted_audio_file is not None and extracted_audio_file.exists():
                extracted_audio_file.unlink()

    except CorruptedMediaError as e:
        # Corrupted files - clear error message about the corruption
        logger.error(f"Corrupted file: {task.name} - {e}")
        return task.file_id, False, str(e)

    except AudioExtractionError as e:
        # Audio extraction failed
        logger.error(f"Audio extraction failed: {task.name} - {e}")
        return task.file_id, False, str(e)

    except NoAudioStreamError as e:
        # File has no audio to transcribe
        logger.error(f"No audio stream: {task.name} - {e}")
        return task.file_id, False, str(e)

    except Exception as e:
        # Catch-all for unexpected errors
        error_msg = f"Failed to process {task.name}"
        logger.exception(error_msg)
        return task.file_id, False, str(e)


def load_catalog(catalog_file: Path) -> list[TranscriptionTask]:
    """Load catalog CSV and filter audio/video files.

    Args:
        catalog_file: Path to catalog.csv file.

    Returns:
        List of TranscriptionTask objects.

    Raises:
        ValueError: If required columns are missing from the catalog.
    """
    tasks: list[TranscriptionTask] = []

    # Required columns
    required_columns = {"gdrive_id", "name", "mime_type"}

    with open(catalog_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate required columns exist
        if reader.fieldnames is None:
            raise ValueError("Catalog file is empty or invalid")

        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"Catalog is missing required columns: {', '.join(missing_columns)}")

        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is line 1)
            try:
                mime_type = row.get("mime_type", "")

                # Filter only audio/video files
                if mime_type not in AUDIO_VIDEO_MIME_TYPES:
                    continue

                # Validate required fields are present
                if not row.get("gdrive_id") or not row.get("name"):
                    logger.warning(f"Skipping row {row_num}: missing gdrive_id or name")
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

                # Parse parents - use shared utility function
                parents_str = row.get("parents", "[]")
                parents = _parse_parents_from_string(parents_str)

                tasks.append(
                    TranscriptionTask(
                        file_id=row["gdrive_id"],
                        name=row["name"],
                        mime_type=mime_type,
                        size_bytes=size_bytes,
                        parents=parents if isinstance(parents, list) else [],
                        web_content_link=row.get("web_content_link", ""),
                        duration_ms=duration_ms,
                    )
                )

            except KeyError as e:
                logger.warning(f"Skipping row {row_num}: missing required field {e}")
                continue
            except Exception as e:
                logger.warning(f"Skipping row {row_num}: {e}")
                continue

    logger.info(f"Loaded {len(tasks)} audio/video files from catalog")
    return tasks


def run_batch_transcription(config: BatchConfig) -> None:
    """Run batch transcription with parallel processing and checkpointing.

    Args:
        config: BatchConfig with all processing parameters.
    """
    # Load results configuration
    results_config = ResultsConfig()
    results_mgr: ResultsManager | None = None

    # Initialize versioned results if enabled
    if results_config.enable_versioning:
        results_mgr = ResultsManager(
            results_config.base_dir,
            PipelineType.TRANSCRIPTION,
            keep_latest_symlinks=results_config.keep_latest_symlinks,
        )
        effective_transcriber_config = TranscriberConfig(
            model_id=config.model_id,
            force_cpu=config.force_cpu,
            quantize=config.quantize,
            language=config.language or "",
        )
        results_mgr.create_run(effective_transcriber_config, input_source=str(config.catalog_file))
        # Override output directory to use versioned path
        effective_output_dir = results_mgr.outputs_dir
        # Use checkpoint file in the run directory
        effective_checkpoint_file = results_mgr.run_dir / "checkpoint.json"
    else:
        effective_output_dir = config.output_dir
        effective_checkpoint_file = config.checkpoint_file

    # Create output directory
    effective_output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = CheckpointManager(effective_checkpoint_file)

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
        if results_mgr is not None:
            results_mgr.update_progress(len(tasks), 0, len(tasks))
            results_mgr.complete_run(success=True)
        return

    # Create a modified config with the effective output directory
    effective_config = BatchConfig(
        catalog_file=config.catalog_file,
        output_dir=effective_output_dir,
        checkpoint_file=effective_checkpoint_file,
        credentials_file=config.credentials_file,
        token_file=config.token_file,
        model_id=config.model_id,
        num_workers=config.num_workers,
        force_cpu=config.force_cpu,
        quantize=config.quantize,
        language=config.language,
    )

    # Determine number of workers
    num_workers = min(config.num_workers, len(remaining_tasks))
    # Only limit workers to CPU count when using CPU mode
    if config.force_cpu and num_workers > mp.cpu_count():
        logger.warning(f"Requested {num_workers} workers but only {mp.cpu_count()} CPUs available")
        num_workers = mp.cpu_count()
    elif num_workers > mp.cpu_count():
        logger.info(
            f"Using {num_workers} workers with GPU processing "
            f"(more than {mp.cpu_count()} CPU cores)"
        )

    logger.info(f"Using {num_workers} parallel workers")

    error_message: str | None = None
    try:
        # Process files in parallel
        if num_workers == 1:
            # Sequential processing for single worker
            for task in remaining_tasks:
                file_id, success, message = transcribe_single_file(task, effective_config)

                if success:
                    checkpoint.mark_completed(file_id)
                    logger.info(f"✓ Completed: {task.name}")
                else:
                    checkpoint.mark_failed(file_id, message)
                    logger.error(f"✗ Failed: {task.name} - {message}")

                completed, total = checkpoint.get_progress()
                logger.info(f"Progress: {completed}/{total} files")

                # Update results manager progress
                if results_mgr is not None:
                    failed_count = len(checkpoint.state.failed_files)
                    results_mgr.update_progress(completed, failed_count, total)
        else:
            # Parallel processing with worker initialization
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp.get_context("forkserver"),
                initializer=_init_worker,
                initargs=(
                    effective_config.model_id,
                    effective_config.force_cpu,
                    effective_config.quantize,
                    effective_config.language,
                ),
            ) as executor:
                # Submit tasks in batches to avoid spawning all at once
                # This prevents resource exhaustion with thousands of pending futures
                batch_size = max(num_workers * 2, 10)
                task_iter = iter(remaining_tasks)
                pending_futures = {}

                # Submit initial batch
                for _ in range(min(batch_size, len(remaining_tasks))):
                    try:
                        task = next(task_iter)
                        future = executor.submit(transcribe_single_file, task, effective_config)
                        pending_futures[future] = task
                    except StopIteration:
                        break

                # Process results and submit new tasks as workers become available
                while pending_futures:
                    # Get the next completed future
                    completed_future = next(as_completed(pending_futures))
                    task = pending_futures.pop(completed_future)

                    try:
                        file_id, success, message = completed_future.result()

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

                    # Update results manager progress
                    if results_mgr is not None:
                        failed_count = len(checkpoint.state.failed_files)
                        results_mgr.update_progress(completed, failed_count, total)

                    # Submit next task to replace the completed one
                    try:
                        next_task = next(task_iter)
                        next_future = executor.submit(
                            transcribe_single_file, next_task, effective_config
                        )
                        pending_futures[next_future] = next_task
                    except StopIteration:
                        pass  # No more tasks to submit

    except Exception as e:
        error_message = str(e)
        logger.exception("Batch transcription failed with exception")
        raise

    finally:
        # Final summary
        completed, total = checkpoint.get_progress()
        failed_count = len(checkpoint.state.failed_files)

        logger.info("=" * 60)
        logger.info("Batch transcription completed!")
        logger.info(f"Total files: {total}")
        logger.info(f"Successfully transcribed: {completed}")
        logger.info(f"Failed: {failed_count}")
        if total == 0:
            logger.info("Success rate: N/A (no files to process)")
        else:
            logger.info(f"Success rate: {completed / total * 100:.1f}%")
        logger.info("=" * 60)

        if failed_count > 0:
            logger.warning("Failed files:")
            for file_id, error in checkpoint.state.failed_files.items():
                logger.warning(f"  - {file_id}: {error}")

        # Complete the versioned run
        if results_mgr is not None:
            success = error_message is None and failed_count < total
            results_mgr.complete_run(success=success, error=error_message)
