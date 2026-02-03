"""Batch QA generation with parallel processing and checkpoint support.

Provides functionality to generate QA pairs from multiple transcription files
with parallel processing, checkpoint/resume capability, and progress tracking.

Supports both standard QA generation and CEP (cognitive scaffolding) generation.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gtranscriber.config import CEPConfig, QAConfig
from gtranscriber.core.checkpoint import CheckpointManager
from gtranscriber.core.llm_client import LLMClient, LLMProvider
from gtranscriber.core.qa_generator import QAGenerator
from gtranscriber.schemas import EnrichedRecord

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Global QA generator instance per worker process
_worker_qa_generator: QAGenerator | None = None


@dataclass
class QAGenerationTask:
    """Task information for QA generation."""

    transcription_file: Path
    gdrive_id: str
    filename: str
    output_file: Path


def _init_qa_worker(provider: str, model_id: str, config_dict: dict) -> None:
    """Initialize QA worker process with generator instance.

    This function is called once per worker process to create the LLM client
    and QA generator, enabling connection pooling across multiple tasks.

    Args:
        provider: LLM provider name (openai, ollama, custom).
        model_id: Model identifier.
        config_dict: QAConfig as dictionary (for pickling).
    """
    global _worker_qa_generator

    # Reconstruct config from dict
    config = QAConfig(**config_dict)

    # Determine base URL
    base_url = config.base_url
    if not base_url and provider == "ollama":
        base_url = config.ollama_url

    # Create LLM client
    llm_client = LLMClient(
        provider=LLMProvider(provider),
        model_id=model_id,
        base_url=base_url,
    )

    # Create QA generator
    _worker_qa_generator = QAGenerator(llm_client, config)

    logger.info(f"QA worker initialized with {provider}/{model_id}")


def generate_qa_for_transcription(
    task: QAGenerationTask,
    config_dict: dict,
) -> tuple[str, bool, str]:
    """Generate QA pairs for a single transcription (worker function).

    Uses the global _worker_qa_generator that was initialized once per worker
    process, enabling connection pooling for HTTP requests to the LLM API.

    Args:
        task: QAGenerationTask with file information.
        config_dict: QAConfig as dictionary (for pickling).

    Returns:
        Tuple of (gdrive_id, success, message).
    """
    global _worker_qa_generator

    try:
        logger.info(f"Processing: {task.filename} ({task.gdrive_id})")

        # For sequential processing, initialize on first use
        if _worker_qa_generator is None:
            config = QAConfig(**config_dict)
            provider = config.provider
            model_id = config.model_id
            base_url = config.base_url or config.ollama_url

            llm_client = LLMClient(
                provider=LLMProvider(provider),
                model_id=model_id,
                base_url=base_url,
            )
            _worker_qa_generator = QAGenerator(llm_client, config)

        # Load transcription
        with open(task.transcription_file, encoding="utf-8") as f:
            data = json.load(f)
            enriched = EnrichedRecord(**data)

        # Generate QA pairs
        qa_record = _worker_qa_generator.generate_qa_pairs(enriched)

        # Save result
        task.output_file.parent.mkdir(parents=True, exist_ok=True)
        qa_record.save(task.output_file)

        logger.info(f"Generated {len(qa_record.qa_pairs)} QA pairs for {task.filename}")
        return task.gdrive_id, True, "Success"

    except ValueError as e:
        # Expected errors (e.g., transcription too short)
        logger.warning(f"Validation error for {task.filename}: {e}")
        return task.gdrive_id, False, str(e)

    except Exception as e:
        # Unexpected errors
        logger.exception(f"Failed to process {task.filename}")
        return task.gdrive_id, False, str(e)


def load_transcription_tasks(input_dir: Path, output_dir: Path) -> list[QAGenerationTask]:
    """Load transcription files and create QA generation tasks.

    Args:
        input_dir: Directory containing EnrichedRecord JSON files.
        output_dir: Directory for QARecord JSON outputs.

    Returns:
        List of QAGenerationTask objects.
    """
    tasks: list[QAGenerationTask] = []

    # Find all transcription JSON files
    transcription_files = list(input_dir.glob("*_transcription.json"))

    if not transcription_files:
        logger.warning(f"No transcription files found in {input_dir}")
        return tasks

    for json_file in transcription_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            gdrive_id = data.get("gdrive_id", "unknown")
            filename = data.get("name", json_file.name)

            # Output filename: {gdrive_id}_qa.json
            output_file = output_dir / f"{gdrive_id}_qa.json"

            tasks.append(
                QAGenerationTask(
                    transcription_file=json_file,
                    gdrive_id=gdrive_id,
                    filename=filename,
                    output_file=output_file,
                )
            )

        except Exception as e:
            logger.warning(f"Skipping invalid file {json_file}: {e}")
            continue

    logger.info(
        f"Loaded {len(tasks)} transcription files from {input_dir} "
        f"({len(transcription_files)} total files found)"
    )
    return tasks


def run_batch_qa_generation(
    input_dir: Path,
    output_dir: Path,
    config: QAConfig,
    num_workers: int = 2,
) -> None:
    """Run batch QA generation with parallel processing and checkpointing.

    Args:
        input_dir: Directory containing transcription JSON files.
        output_dir: Directory for QA dataset outputs.
        config: QA generation configuration.
        num_workers: Number of parallel workers.
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize checkpoint
    checkpoint_file = output_dir / "qa_checkpoint.json"
    checkpoint = CheckpointManager(checkpoint_file)

    # Load tasks
    all_tasks = load_transcription_tasks(input_dir, output_dir)

    if not all_tasks:
        logger.warning("No tasks to process")
        return

    # Filter out already completed files
    remaining_tasks = [t for t in all_tasks if not checkpoint.is_completed(t.gdrive_id)]

    # Update checkpoint with total
    checkpoint.set_total_files(len(all_tasks))

    logger.info(f"Total files: {len(all_tasks)}")
    logger.info(f"Already completed: {len(all_tasks) - len(remaining_tasks)}")
    logger.info(f"Remaining to process: {len(remaining_tasks)}")

    if not remaining_tasks:
        logger.info("All files already processed!")
        return

    # Determine number of workers
    num_workers = min(num_workers, len(remaining_tasks))

    # For CPU-bound LLM work, limiting to CPU count is less critical
    # (the bottleneck is network I/O to LLM API, not CPU)
    if num_workers > mp.cpu_count():
        logger.info(
            f"Using {num_workers} workers (more than {mp.cpu_count()} CPU cores). "
            f"This is fine for I/O-bound LLM API calls."
        )

    logger.info(f"Using {num_workers} parallel workers")

    # Convert config to dict for pickling
    config_dict = config.model_dump()

    # Process files
    if num_workers == 1:
        # Sequential processing
        for task in remaining_tasks:
            gdrive_id, success, message = generate_qa_for_transcription(task, config_dict)

            if success:
                checkpoint.mark_completed(gdrive_id)
                logger.info(f"✓ Completed: {task.filename}")
            else:
                checkpoint.mark_failed(gdrive_id, message)
                logger.error(f"✗ Failed: {task.filename} - {message}")

            completed, total = checkpoint.get_progress()
            logger.info(f"Progress: {completed}/{total} files")

    else:
        # Parallel processing with worker initialization
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_qa_worker,
            initargs=(config.provider, config.model_id, config_dict),
        ) as executor:
            # Batched submission to prevent memory issues with large task lists
            batch_size = max(num_workers * 2, 10)
            task_iter = iter(remaining_tasks)
            pending_futures = {}

            # Submit initial batch
            for _ in range(min(batch_size, len(remaining_tasks))):
                try:
                    task = next(task_iter)
                    future = executor.submit(generate_qa_for_transcription, task, config_dict)
                    pending_futures[future] = task
                except StopIteration:
                    break

            # Process results and submit new tasks as workers become available
            while pending_futures:
                # Get next completed future
                completed_future = next(as_completed(pending_futures))
                task = pending_futures.pop(completed_future)

                try:
                    gdrive_id, success, message = completed_future.result()

                    if success:
                        checkpoint.mark_completed(gdrive_id)
                        logger.info(f"✓ Completed: {task.filename}")
                    else:
                        checkpoint.mark_failed(gdrive_id, message)
                        logger.error(f"✗ Failed: {task.filename} - {message}")

                except Exception as e:
                    logger.exception(f"Task failed with exception: {task.filename}")
                    checkpoint.mark_failed(task.gdrive_id, str(e))

                completed, total = checkpoint.get_progress()
                logger.info(f"Progress: {completed}/{total} files")

                # Submit next task to replace the completed one
                try:
                    next_task = next(task_iter)
                    next_future = executor.submit(
                        generate_qa_for_transcription, next_task, config_dict
                    )
                    pending_futures[next_future] = next_task
                except StopIteration:
                    pass  # No more tasks to submit

    # Final summary
    completed, total = checkpoint.get_progress()
    failed_count = len(checkpoint.state.failed_files)

    logger.info("=" * 60)
    logger.info("Batch QA generation completed!")
    logger.info(f"Total files: {total}")
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


# =============================================================================
# CEP (Cognitive Elicitation Pipeline) Batch Processing
# =============================================================================

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

    from gtranscriber.core.cep import CEPQAGenerator

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
        Tuple of (gdrive_id, success, message).
    """
    global _worker_cep_generator

    try:
        logger.info(f"Processing (CEP): {task.filename} ({task.gdrive_id})")

        # For sequential processing, initialize on first use
        if _worker_cep_generator is None:
            from gtranscriber.core.cep import CEPQAGenerator

            qa_config = QAConfig(**qa_config_dict)
            cep_config = CEPConfig(**cep_config_dict)
            provider = qa_config.provider
            model_id = qa_config.model_id
            base_url = qa_config.base_url or qa_config.ollama_url

            llm_client = LLMClient(
                provider=LLMProvider(provider),
                model_id=model_id,
                base_url=base_url,
            )

            # Create validator client if enabled
            validator_client = None
            if cep_config.enable_validation:
                validator_client = LLMClient(
                    provider=LLMProvider(cep_config.validator_provider),
                    model_id=cep_config.validator_model_id,
                    base_url=base_url,
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
        return task.gdrive_id, True, "Success"

    except ValueError as e:
        logger.warning(f"Validation error for {task.filename}: {e}")
        return task.gdrive_id, False, str(e)

    except Exception as e:
        logger.exception(f"Failed to process {task.filename}")
        return task.gdrive_id, False, str(e)


def run_batch_pec_generation(
    input_dir: Path,
    output_dir: Path,
    qa_config: QAConfig,
    cep_config: CEPConfig,
    num_workers: int = 2,
) -> None:
    """Run batch CEP QA generation with parallel processing and checkpointing.

    Args:
        input_dir: Directory containing transcription JSON files.
        output_dir: Directory for CEP QA dataset outputs.
        qa_config: QA generation configuration.
        cep_config: CEP configuration.
        num_workers: Number of parallel workers.
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize checkpoint
    checkpoint_file = output_dir / "cep_checkpoint.json"
    checkpoint = CheckpointManager(checkpoint_file)

    # Load tasks (reuse existing function, but change output suffix)
    all_tasks = []
    transcription_files = list(input_dir.glob("*_transcription.json"))

    for json_file in transcription_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            gdrive_id = data.get("gdrive_id", "unknown")
            filename = data.get("name", json_file.name)

            # Output filename: {gdrive_id}_cep_qa.json
            output_file = output_dir / f"{gdrive_id}_cep_qa.json"

            all_tasks.append(
                QAGenerationTask(
                    transcription_file=json_file,
                    gdrive_id=gdrive_id,
                    filename=filename,
                    output_file=output_file,
                )
            )

        except Exception as e:
            logger.warning(f"Skipping invalid file {json_file}: {e}")
            continue

    if not all_tasks:
        logger.warning("No tasks to process")
        return

    # Filter out already completed files
    remaining_tasks = [t for t in all_tasks if not checkpoint.is_completed(t.gdrive_id)]

    # Update checkpoint with total
    checkpoint.set_total_files(len(all_tasks))

    logger.info(f"Total files: {len(all_tasks)}")
    logger.info(f"Already completed: {len(all_tasks) - len(remaining_tasks)}")
    logger.info(f"Remaining to process: {len(remaining_tasks)}")

    if not remaining_tasks:
        logger.info("All files already processed!")
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

    # Process files
    if num_workers == 1:
        # Sequential processing
        for task in remaining_tasks:
            gdrive_id, success, message = generate_cep_qa_for_transcription(
                task, qa_config_dict, cep_config_dict
            )

            if success:
                checkpoint.mark_completed(gdrive_id)
                logger.info(f"✓ Completed: {task.filename}")
            else:
                checkpoint.mark_failed(gdrive_id, message)
                logger.error(f"✗ Failed: {task.filename} - {message}")

            completed, total = checkpoint.get_progress()
            logger.info(f"Progress: {completed}/{total} files")

    else:
        # Parallel processing with worker initialization
        with ProcessPoolExecutor(
            max_workers=num_workers,
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
                    gdrive_id, success, message = completed_future.result()

                    if success:
                        checkpoint.mark_completed(gdrive_id)
                        logger.info(f"✓ Completed: {task.filename}")
                    else:
                        checkpoint.mark_failed(gdrive_id, message)
                        logger.error(f"✗ Failed: {task.filename} - {message}")

                except Exception as e:
                    logger.exception(f"Task failed with exception: {task.filename}")
                    checkpoint.mark_failed(task.gdrive_id, str(e))

                completed, total = checkpoint.get_progress()
                logger.info(f"Progress: {completed}/{total} files")

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

    # Final summary
    completed, total = checkpoint.get_progress()
    failed_count = len(checkpoint.state.failed_files)

    logger.info("=" * 60)
    logger.info("Batch CEP QA generation completed!")
    logger.info(f"Total files: {total}")
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
