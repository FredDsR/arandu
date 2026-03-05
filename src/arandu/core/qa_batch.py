"""Batch CEP QA generation with parallel processing and checkpoint support.

Provides functionality to generate CEP (cognitive scaffolding) QA pairs from
multiple transcription files with parallel processing, checkpoint/resume
capability, and progress tracking.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path  # noqa: TC003 — Pydantic needs Path at runtime

from pydantic import BaseModel

from arandu.config import CEPConfig, QAConfig, ResultsConfig
from arandu.schemas import EnrichedRecord, PipelineType
from arandu.shared.checkpoint import CheckpointManager
from arandu.shared.llm_client import LLMClient, LLMProvider
from arandu.shared.results_manager import ResultsManager

logger = logging.getLogger(__name__)


class QAGenerationTask(BaseModel):
    """Task information for QA generation."""

    transcription_file: Path
    file_id: str
    filename: str
    output_file: Path


class TaskLoadResult(BaseModel):
    """Result of loading transcription tasks with filtering metadata."""

    tasks: list[QAGenerationTask]
    total_found: int
    skipped_invalid: int


def _resolve_transcription_dir(
    input_dir: Path,
    pipeline_id: str | None = None,
) -> Path:
    """Resolve the directory containing transcription JSON files.

    If ``pipeline_id`` is given, looks up
    ``input_dir/{pipeline_id}/transcription/outputs`` directly.
    If ``input_dir`` already contains ``*_transcription.json`` files it is
    returned as-is.  Otherwise, the directory is treated as a versioned
    results base directory and the latest transcription outputs are resolved
    via :pyclass:`ResultsManager`.

    Args:
        input_dir: Directory provided by the caller (may be the base results
            directory or a direct path to transcription outputs).
        pipeline_id: Optional pipeline ID for direct resolution.

    Returns:
        The directory that contains ``*_transcription.json`` files.
    """
    # Direct resolution by pipeline_id
    if pipeline_id is not None:
        resolved = ResultsManager.resolve_outputs(
            input_dir, pipeline_id, PipelineType.TRANSCRIPTION
        )
        if resolved is not None:
            logger.info(f"Resolved transcription outputs for pipeline {pipeline_id}: {resolved}")
            return resolved

    # Fast path: input_dir already contains transcription files
    if list(input_dir.glob("*_transcription.json")):
        return input_dir

    # Treat input_dir as a versioned results base directory
    resolved = ResultsManager.resolve_latest_outputs(input_dir, PipelineType.TRANSCRIPTION)
    if resolved is not None:
        logger.info(f"Resolved transcription outputs from versioned results: {resolved}")
        return resolved

    # Return original dir so callers get the standard "no files found" warning
    return input_dir


def load_transcription_tasks(
    input_dir: Path,
    output_dir: Path,
    pipeline_id: str | None = None,
    output_suffix: str = "_cep_qa.json",
) -> TaskLoadResult:
    """Load transcription files and create QA generation tasks.

    Transcriptions where ``is_valid`` is explicitly ``False`` are skipped.
    Transcriptions without the field or with ``is_valid=None`` are included.

    Args:
        input_dir: Directory containing EnrichedRecord JSON files, or the
            base versioned results directory.
        output_dir: Directory for QA output JSON files.
        pipeline_id: Optional pipeline ID for direct transcription resolution.
        output_suffix: Suffix for output filenames (e.g. ``"_cep_qa.json"``).

    Returns:
        A TaskLoadResult containing the tasks list and filtering metadata
        (total files found and number skipped as invalid).
    """
    tasks: list[QAGenerationTask] = []

    # Resolve versioned directory layout when needed
    effective_input_dir = _resolve_transcription_dir(input_dir, pipeline_id=pipeline_id)

    # Find all transcription JSON files
    transcription_files = list(effective_input_dir.glob("*_transcription.json"))

    if not transcription_files:
        logger.warning(f"No transcription files found in {effective_input_dir}")
        return TaskLoadResult(tasks=tasks, total_found=0, skipped_invalid=0)

    skipped_invalid = 0
    for json_file in transcription_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Skip transcriptions that failed quality validation
            if data.get("is_valid") is False:
                skipped_invalid += 1
                logger.info(f"Skipping invalid transcription: {json_file.name}")
                continue

            file_id = data.get("file_id") or data.get("gdrive_id", "unknown")
            filename = data.get("name", json_file.name)

            output_file = output_dir / f"{file_id}{output_suffix}"

            tasks.append(
                QAGenerationTask(
                    transcription_file=json_file,
                    file_id=file_id,
                    filename=filename,
                    output_file=output_file,
                )
            )

        except Exception as e:
            logger.warning(f"Skipping invalid file {json_file}: {e}")
            continue

    logger.info(
        f"Loaded {len(tasks)} transcription files from {input_dir} "
        f"({len(transcription_files)} total files found, "
        f"{skipped_invalid} skipped as invalid)"
    )
    return TaskLoadResult(
        tasks=tasks,
        total_found=len(transcription_files),
        skipped_invalid=skipped_invalid,
    )


# Global CEP generator instance per worker process
_worker_cep_generator = None


def _init_cep_worker(
    provider: str,
    model_id: str,
    qa_config_dict: dict,
    cep_config_dict: dict,
    validator_provider: str | None,
    validator_model_id: str | None,
) -> None:
    """Initialize CEP worker process with generator instance.

    Args:
        provider: LLM provider name.
        model_id: Model identifier.
        qa_config_dict: QAConfig as dictionary.
        cep_config_dict: CEPConfig as dictionary.
        validator_provider: Validator LLM provider (if validation enabled).
        validator_model_id: Validator model identifier (if validation enabled).
    """
    global _worker_cep_generator

    from arandu.core.cep import CEPQAGenerator

    # Reconstruct configs
    qa_config = QAConfig(**qa_config_dict)
    cep_config = CEPConfig(**cep_config_dict)

    # Determine base URL
    base_url = qa_config.base_url
    if not base_url and provider == "ollama":
        base_url = qa_config.ollama_url

    # Create main LLM client
    llm_client = LLMClient(
        provider=LLMProvider(provider),
        model_id=model_id,
        base_url=base_url,
    )

    # Create validator client if validation is enabled
    validator_client = None
    if cep_config.enable_validation and validator_provider and validator_model_id:
        validator_base_url = base_url
        if validator_provider == "ollama":
            validator_base_url = qa_config.ollama_url

        validator_client = LLMClient(
            provider=LLMProvider(validator_provider),
            model_id=validator_model_id,
            base_url=validator_base_url,
        )

    # Create CEP generator
    _worker_cep_generator = CEPQAGenerator(
        llm_client=llm_client,
        qa_config=qa_config,
        cep_config=cep_config,
        validator_client=validator_client,
    )

    logger.info(f"CEP worker initialized with {provider}/{model_id}")


def generate_cep_qa_for_transcription(
    task: QAGenerationTask,
    qa_config_dict: dict,
    cep_config_dict: dict,
) -> tuple[str, bool, str]:
    """Generate CEP QA pairs for a single transcription (worker function).

    Args:
        task: QAGenerationTask with file information.
        qa_config_dict: QAConfig as dictionary.
        cep_config_dict: CEPConfig as dictionary.

    Returns:
        Tuple of (file_id, success, message).
    """
    global _worker_cep_generator

    try:
        logger.info(f"Processing (CEP): {task.filename} ({task.file_id})")

        # For sequential processing, initialize on first use
        if _worker_cep_generator is None:
            from arandu.core.cep import CEPQAGenerator

            qa_config = QAConfig(**qa_config_dict)
            cep_config = CEPConfig(**cep_config_dict)
            provider = qa_config.provider
            model_id = qa_config.model_id
            # Only fall back to ollama_url when provider is ollama
            base_url = qa_config.base_url
            if not base_url and provider == "ollama":
                base_url = qa_config.ollama_url

            llm_client = LLMClient(
                provider=LLMProvider(provider),
                model_id=model_id,
                base_url=base_url,
            )

            # Create validator client if enabled
            validator_client = None
            if cep_config.enable_validation:
                # Choose validator base URL based on validator provider
                if cep_config.validator_provider == LLMProvider.OLLAMA.value:
                    validator_base_url = qa_config.ollama_url
                else:
                    validator_base_url = qa_config.base_url

                validator_client = LLMClient(
                    provider=LLMProvider(cep_config.validator_provider),
                    model_id=cep_config.validator_model_id,
                    base_url=validator_base_url,
                )

            _worker_cep_generator = CEPQAGenerator(
                llm_client=llm_client,
                qa_config=qa_config,
                cep_config=cep_config,
                validator_client=validator_client,
            )

        # Load transcription
        with open(task.transcription_file, encoding="utf-8") as f:
            data = json.load(f)
            enriched = EnrichedRecord(**data)

        # Generate CEP QA pairs
        qa_record = _worker_cep_generator.generate_qa_pairs(enriched)

        # Save result
        task.output_file.parent.mkdir(parents=True, exist_ok=True)
        qa_record.save(task.output_file)

        logger.info(
            f"Generated {len(qa_record.qa_pairs)} CEP QA pairs for {task.filename} "
            f"(validated: {qa_record.validated_pairs})"
        )
        return task.file_id, True, "Success"

    except ValueError as e:
        logger.warning(f"Validation error for {task.filename}: {e}")
        return task.file_id, False, str(e)

    except Exception as e:
        logger.exception(f"Failed to process {task.filename}")
        return task.file_id, False, str(e)


def run_batch_cep_generation(
    input_dir: Path,
    output_dir: Path,
    qa_config: QAConfig,
    cep_config: CEPConfig,
    num_workers: int = 2,
    pipeline_id: str | None = None,
) -> None:
    """Run batch CEP QA generation with parallel processing and checkpointing.

    Transcriptions that failed quality validation (``is_valid=False``) are
    automatically filtered out during task loading.  The number of skipped
    files is reported in both the initial info logs and the final summary.

    Args:
        input_dir: Directory containing transcription JSON files.
        output_dir: Directory for CEP QA dataset outputs.
        qa_config: QA generation configuration.
        cep_config: CEP configuration.
        num_workers: Number of parallel workers.
        pipeline_id: Optional pipeline ID for the ID-first results layout.
    """
    # Load results configuration
    results_config = ResultsConfig()
    results_mgr: ResultsManager | None = None

    # Initialize versioned results if enabled
    if results_config.enable_versioning:
        results_mgr = ResultsManager(
            results_config.base_dir,
            PipelineType.CEP,
            pipeline_id=pipeline_id,
        )
        results_mgr.create_run(
            qa_config,
            input_source=str(input_dir),
            checkpoint_filename="cep_checkpoint.json",
        )
        # Supplement config snapshot with CEP settings for reproducibility
        if results_mgr.metadata.config:
            results_mgr.metadata.config.config_values["cep_config"] = cep_config.model_dump(
                mode="json"
            )
            results_mgr.metadata.config.config_type = "QAConfig+CEPConfig"
            results_mgr.metadata.save(results_mgr.run_dir / "run_metadata.json")
        # Override output directory to use versioned path
        effective_output_dir = results_mgr.outputs_dir
        # Use checkpoint file in the run directory
        effective_checkpoint_file = results_mgr.run_dir / "cep_checkpoint.json"
    else:
        effective_output_dir = output_dir
        effective_checkpoint_file = output_dir / "cep_checkpoint.json"

    # Create output directory
    effective_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize checkpoint
    checkpoint = CheckpointManager(effective_checkpoint_file)

    # Load tasks
    load_result = load_transcription_tasks(input_dir, effective_output_dir, pipeline_id=pipeline_id)
    all_tasks = load_result.tasks
    skipped_invalid = load_result.skipped_invalid

    if not all_tasks:
        logger.warning("No tasks to process")
        if results_mgr is not None:
            results_mgr.complete_run(success=True)
        return

    # Filter out already completed files
    remaining_tasks = [t for t in all_tasks if not checkpoint.is_completed(t.file_id)]

    # Update checkpoint with total
    checkpoint.set_total_files(len(all_tasks))

    logger.info(f"Total files: {len(all_tasks)}")
    if skipped_invalid > 0:
        logger.info(f"Skipped invalid transcriptions: {skipped_invalid}")
    logger.info(f"Already completed: {len(all_tasks) - len(remaining_tasks)}")
    logger.info(f"Remaining to process: {len(remaining_tasks)}")

    if not remaining_tasks:
        logger.info("All files already processed!")
        if results_mgr is not None:
            results_mgr.update_progress(len(all_tasks), 0, len(all_tasks))
            results_mgr.complete_run(success=True)
        return

    # Determine number of workers
    num_workers = min(num_workers, len(remaining_tasks))

    if num_workers > mp.cpu_count():
        logger.info(
            f"Using {num_workers} workers (more than {mp.cpu_count()} CPU cores). "
            f"This is fine for I/O-bound LLM API calls."
        )

    logger.info(f"Using {num_workers} parallel workers")

    # Convert configs to dict for pickling
    qa_config_dict = qa_config.model_dump()
    cep_config_dict = cep_config.model_dump()

    # Validator info for worker initialization
    validator_provider = cep_config.validator_provider if cep_config.enable_validation else None
    validator_model_id = cep_config.validator_model_id if cep_config.enable_validation else None

    error_message: str | None = None
    try:
        # Process files
        if num_workers == 1:
            # Sequential processing
            for task in remaining_tasks:
                file_id, success, message = generate_cep_qa_for_transcription(
                    task, qa_config_dict, cep_config_dict
                )

                if success:
                    checkpoint.mark_completed(file_id)
                    logger.info(f"✓ Completed: {task.filename}")
                else:
                    checkpoint.mark_failed(file_id, message)
                    logger.error(f"✗ Failed: {task.filename} - {message}")

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
                initializer=_init_cep_worker,
                initargs=(
                    qa_config.provider,
                    qa_config.model_id,
                    qa_config_dict,
                    cep_config_dict,
                    validator_provider,
                    validator_model_id,
                ),
            ) as executor:
                # Batched submission
                batch_size = max(num_workers * 2, 10)
                task_iter = iter(remaining_tasks)
                pending_futures = {}

                # Submit initial batch
                for _ in range(min(batch_size, len(remaining_tasks))):
                    try:
                        task = next(task_iter)
                        future = executor.submit(
                            generate_cep_qa_for_transcription,
                            task,
                            qa_config_dict,
                            cep_config_dict,
                        )
                        pending_futures[future] = task
                    except StopIteration:
                        break

                # Process results and submit new tasks
                while pending_futures:
                    completed_future = next(as_completed(pending_futures))
                    task = pending_futures.pop(completed_future)

                    try:
                        file_id, success, message = completed_future.result()

                        if success:
                            checkpoint.mark_completed(file_id)
                            logger.info(f"✓ Completed: {task.filename}")
                        else:
                            checkpoint.mark_failed(file_id, message)
                            logger.error(f"✗ Failed: {task.filename} - {message}")

                    except Exception as e:
                        logger.exception(f"Task failed with exception: {task.filename}")
                        checkpoint.mark_failed(task.file_id, str(e))

                    completed, total = checkpoint.get_progress()
                    logger.info(f"Progress: {completed}/{total} files")

                    # Update results manager progress
                    if results_mgr is not None:
                        failed_count = len(checkpoint.state.failed_files)
                        results_mgr.update_progress(completed, failed_count, total)

                    # Submit next task
                    try:
                        next_task = next(task_iter)
                        next_future = executor.submit(
                            generate_cep_qa_for_transcription,
                            next_task,
                            qa_config_dict,
                            cep_config_dict,
                        )
                        pending_futures[next_future] = next_task
                    except StopIteration:
                        pass

    except Exception as e:
        error_message = str(e)
        logger.exception("Batch CEP QA generation failed with exception")
        raise

    finally:
        # Final summary
        completed, total = checkpoint.get_progress()
        failed_count = len(checkpoint.state.failed_files)

        logger.info("=" * 60)
        logger.info("Batch CEP QA generation completed!")
        logger.info(f"Total files: {total}")
        if skipped_invalid > 0:
            logger.info(f"Skipped invalid transcriptions: {skipped_invalid}")
        logger.info(f"Successfully processed: {completed}")
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
