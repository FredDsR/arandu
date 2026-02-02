# Batch Processing Patterns - Analysis for Phase 2 Implementation

**Purpose**: Understand existing patterns in G-Transcriber to ensure consistency when implementing QA generation, KG construction, and evaluation pipelines.

**Status**: Phase 1 ✅ Complete | Phase 2 🔄 Ready to implement

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Checkpoint System Pattern](#checkpoint-system-pattern)
3. [Batch Processing Pattern](#batch-processing-pattern)
4. [Worker Initialization Pattern](#worker-initialization-pattern)
5. [CLI Command Pattern](#cli-command-pattern)
6. [Error Handling Pattern](#error-handling-pattern)
7. [Applying Patterns to Phase 2](#applying-patterns-to-phase-2)

---

## Architecture Overview

### Core Components

The existing transcription batch processing system consists of:

```
┌─────────────────────────────────────────────────────────────┐
│                    Batch Processing System                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ CLI Command  │→ │ Batch Loader │→ │   Workers    │     │
│  │  (main.py)   │  │  (batch.py)  │  │(parallel exec)│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         ↓                  ↓                   ↓            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Config    │  │  Checkpoint  │  │    Engine    │     │
│  │  (config.py) │  │(checkpoint.py)│  │  (engine.py) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Fault Tolerance**: Checkpoint system enables resume after failure
3. **Parallel Processing**: Uses `ProcessPoolExecutor` for true parallelism
4. **Configuration-Driven**: All settings via Pydantic with env var support
5. **Type Safety**: Full type annotations throughout
6. **Error Handling**: Graceful degradation with detailed logging

---

## Checkpoint System Pattern

### Location
`src/gtranscriber/core/checkpoint.py` (154 lines)

### Purpose
Track progress of batch processing jobs and enable resumption after interruption.

### Key Classes

#### 1. CheckpointState (Pydantic Model)

```python
class CheckpointState(BaseModel):
    """State tracking for batch processing checkpoint."""

    completed_files: set[str] = Field(default_factory=set)
    failed_files: dict[str, str] = Field(default_factory=dict)
    total_files: int = Field(0)
    started_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
```

**Key Features**:
- Uses `set` for completed files (O(1) lookup)
- Tracks failed files with error messages
- Timestamps for monitoring
- Serializable to JSON

#### 2. CheckpointManager (Class)

```python
class CheckpointManager:
    """Manages checkpoint state for batch processing."""

    def __init__(self, checkpoint_file: Path) -> None
    def mark_completed(self, file_id: str) -> None
    def mark_failed(self, file_id: str, error: str) -> None
    def is_completed(self, file_id: str) -> bool
    def get_progress(self) -> tuple[int, int]
```

**Key Methods**:
- `_load()` - Loads checkpoint from JSON, handles corruption gracefully
- `save()` - Persists state after every update
- `mark_completed()` - Atomically marks file as done and saves
- `mark_failed()` - Records failure with error message
- `is_completed()` - Fast O(1) lookup for filtering

### Usage Pattern

```python
# Initialize checkpoint
checkpoint = CheckpointManager(Path("results/checkpoint.json"))

# Filter already completed tasks
remaining_tasks = [t for t in all_tasks if not checkpoint.is_completed(t.id)]

# Update total count
checkpoint.set_total_files(len(all_tasks))

# After processing each file
if success:
    checkpoint.mark_completed(file_id)
else:
    checkpoint.mark_failed(file_id, error_message)

# Get progress
completed, total = checkpoint.get_progress()
```

### Important Notes

1. **Auto-save**: Every `mark_completed()` and `mark_failed()` triggers save
2. **Thread-safe**: Can be called from multiple processes (file-based locking)
3. **Corruption handling**: Invalid checkpoint files fall back to fresh start
4. **Set conversion**: Converts list ↔ set for JSON serialization

---

## Batch Processing Pattern

### Location
`src/gtranscriber/core/batch.py` (563 lines)

### Purpose
Orchestrate parallel processing of multiple files with checkpoint integration.

### Key Components

#### 1. Configuration Dataclass

```python
@dataclass
class BatchConfig:
    """Configuration for batch processing."""

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
    def from_transcriber_config(cls, ...) -> BatchConfig:
        """Factory method to create from main config."""
```

**Pattern**: Use `@dataclass` for configuration objects that combine:
- CLI arguments
- Environment variables
- Default values

#### 2. Task Dataclass

```python
@dataclass
class TranscriptionTask:
    """Task information for processing."""

    file_id: str
    name: str
    mime_type: str
    size_bytes: int | None
    parents: list[str]
    web_content_link: str
    duration_ms: int | None
```

**Pattern**: Simple dataclass for task parameters (no Pydantic validation needed)

#### 3. Catalog Loader

```python
def load_catalog(catalog_file: Path) -> list[TranscriptionTask]:
    """Load catalog CSV and filter relevant files.

    Returns:
        List of task objects ready for processing.
    """
    tasks: list[TranscriptionTask] = []

    with open(catalog_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate required columns
        required_columns = {"gdrive_id", "name", "mime_type"}
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Parse and filter rows
        for row in reader:
            if row["mime_type"] in VALID_MIME_TYPES:
                tasks.append(TranscriptionTask(...))

    return tasks
```

**Pattern**:
- Validate schema upfront
- Filter irrelevant entries early
- Log warnings for skipped rows
- Return strongly-typed objects

#### 4. Worker Function

```python
def transcribe_single_file(
    task: TranscriptionTask,
    config: BatchConfig,
) -> tuple[str, bool, str]:
    """Process a single file (worker function).

    Returns:
        Tuple of (file_id, success, message).
    """
    try:
        # Use global _worker_engine (initialized once per process)
        global _worker_engine

        # 1. Download/prepare input
        # 2. Process with engine
        # 3. Save output
        # 4. Cleanup temporary files

        return task.file_id, True, "Success"

    except SpecificError as e:
        logger.error(f"Specific error: {e}")
        return task.file_id, False, str(e)

    except Exception as e:
        logger.exception(f"Unexpected error")
        return task.file_id, False, str(e)
```

**Pattern**:
- Accept task + config, return (id, success, message)
- Use global worker engine (initialized once)
- Clean up resources in `finally` block
- Catch specific exceptions before generic
- Always return triple (never raise from worker)

#### 5. Main Orchestrator

```python
def run_batch_transcription(config: BatchConfig) -> None:
    """Run batch processing with parallel workers and checkpointing."""

    # 1. Setup
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointManager(config.checkpoint_file)

    # 2. Load tasks
    all_tasks = load_catalog(config.catalog_file)
    remaining_tasks = [t for t in all_tasks if not checkpoint.is_completed(t.id)]
    checkpoint.set_total_files(len(all_tasks))

    # 3. Determine worker count
    num_workers = min(config.num_workers, len(remaining_tasks))
    if config.force_cpu:
        num_workers = min(num_workers, mp.cpu_count())

    # 4. Process (single vs parallel)
    if num_workers == 1:
        # Sequential processing
        for task in remaining_tasks:
            file_id, success, message = transcribe_single_file(task, config)
            if success:
                checkpoint.mark_completed(file_id)
            else:
                checkpoint.mark_failed(file_id, message)
    else:
        # Parallel processing with batched submission
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(config.model_id, ...),
        ) as executor:
            # Batch submission pattern (prevents memory issues)
            batch_size = max(num_workers * 2, 10)
            task_iter = iter(remaining_tasks)
            pending_futures = {}

            # Submit initial batch
            for _ in range(min(batch_size, len(remaining_tasks))):
                task = next(task_iter)
                future = executor.submit(transcribe_single_file, task, config)
                pending_futures[future] = task

            # Process and submit new tasks as workers finish
            while pending_futures:
                completed_future = next(as_completed(pending_futures))
                task = pending_futures.pop(completed_future)

                file_id, success, message = completed_future.result()

                if success:
                    checkpoint.mark_completed(file_id)
                else:
                    checkpoint.mark_failed(file_id, message)

                # Submit next task
                try:
                    next_task = next(task_iter)
                    next_future = executor.submit(...)
                    pending_futures[next_future] = next_task
                except StopIteration:
                    pass

    # 5. Summary
    completed, total = checkpoint.get_progress()
    logger.info(f"Completed: {completed}/{total}")
```

**Key Patterns**:
- **Batched submission**: Don't submit all futures at once (memory)
- **Dynamic replacement**: Submit new task when one completes
- **Sequential fallback**: Single worker runs without executor overhead
- **Worker count limits**: Respect CPU count for CPU mode
- **Progress logging**: After each completion

---

## Worker Initialization Pattern

### Global Engine Pattern

```python
# Global variable (one per process)
_worker_engine: WhisperEngine | None = None

def _init_worker(
    model_id: str,
    force_cpu: bool,
    quantize: bool,
    language: str | None
) -> None:
    """Initialize worker process with engine instance.

    Called once per worker process to load the model.
    """
    global _worker_engine
    _worker_engine = WhisperEngine(
        model_id=model_id,
        force_cpu=force_cpu,
        quantize=quantize,
        language=language,
    )
    logger.info(f"Worker initialized with model {model_id}")

def transcribe_single_file(task, config):
    """Worker function uses pre-initialized global engine."""
    global _worker_engine

    # For sequential processing, initialize on first use
    if _worker_engine is None:
        _worker_engine = WhisperEngine(...)

    result = _worker_engine.transcribe(file)
    return file_id, True, "Success"
```

**Why This Pattern?**
1. **Efficiency**: Load heavy model once per process, not per file
2. **Memory**: Each process has its own model instance
3. **Parallelism**: True parallelism (not limited by GIL)
4. **Fallback**: Sequential mode initializes on first use

**Usage with ProcessPoolExecutor**:
```python
with ProcessPoolExecutor(
    max_workers=num_workers,
    initializer=_init_worker,  # Called once per worker process
    initargs=(model_id, force_cpu, quantize, language),
) as executor:
    # Submit tasks...
```

---

## CLI Command Pattern

### Location
`src/gtranscriber/main.py` - `batch_transcribe()` command (lines 475-625)

### Structure

```python
@app.command()
def batch_transcribe(
    # 1. REQUIRED ARGUMENTS
    catalog: Annotated[
        Path,
        typer.Argument(
            help="Path to catalog CSV file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],

    # 2. OPTIONAL PARAMETERS WITH DEFAULTS FROM CONFIG
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o",
            help="Output directory for results.",
        ),
    ] = Path(_config.results_dir),  # Default from config

    model_id: Annotated[
        str,
        typer.Option(
            "--model-id", "-m",
            help="Model ID. Can be set via GTRANSCRIBER_MODEL_ID env var.",
        ),
    ] = _config.model_id,  # Default from config

    # ... more options ...

) -> None:
    """Batch transcribe files from catalog.

    Detailed description with:
    - What the command does
    - Input format requirements
    - Output format
    - Features (checkpoint, resume, etc.)
    """
    # 1. VALIDATION
    if not credentials.exists():
        print_error(f"Credentials file not found: {credentials}")
        raise typer.Exit(code=1)

    if workers < 1:
        print_error("Workers must be at least 1")
        raise typer.Exit(code=1)

    # 2. CREATE CONFIG OBJECT
    config = BatchConfig(
        catalog_file=catalog,
        output_dir=output_dir,
        # ... all parameters ...
    )

    # 3. DISPLAY CONFIGURATION
    console.print("\n[bold]Configuration[/bold]\n")
    console.print(f"[cyan]Catalog:[/cyan] {catalog}")
    console.print(f"[cyan]Workers:[/cyan] {workers}")
    console.print()

    # 4. RUN WITH ERROR HANDLING
    try:
        run_batch_transcription(config)
        print_success("Completed!")

    except Exception as e:
        print_error(f"Failed: {e}")
        raise typer.Exit(code=1) from e
```

### Key Patterns

1. **Annotated types**: Use `Annotated[Type, typer.Option(...)]`
2. **Config defaults**: Parameters default to `_config.attribute`
3. **Validation**: Check inputs before processing
4. **Display config**: Show user what will run
5. **Error handling**: Catch, log, and exit with code 1
6. **Rich output**: Use `print_info`, `print_success`, `print_error`

### Module-Level Config

```python
# At top of main.py
_config = TranscriberConfig()  # Loaded once at import

# Use in command defaults
def command(
    param: str = _config.param_name
):
    ...
```

---

## Error Handling Pattern

### Hierarchy of Exception Handling

```python
def worker_function(task, config):
    """Worker always returns (id, success, message) - never raises."""
    try:
        # Main processing logic
        result = process(task)
        return task.id, True, "Success"

    except SpecificError1 as e:
        # Handle known error type 1
        logger.error(f"Error type 1: {e}")
        return task.id, False, str(e)

    except SpecificError2 as e:
        # Handle known error type 2
        logger.error(f"Error type 2: {e}")
        return task.id, False, str(e)

    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(f"Unexpected error")
        return task.id, False, str(e)

    finally:
        # Always cleanup resources
        cleanup_temp_files()
```

### Cleanup Pattern

```python
def process_file(file_path: Path) -> Result:
    temp_file = create_temp_file(suffix=".tmp")
    extracted_file: Path | None = None

    try:
        # Processing logic
        if needs_extraction:
            extracted_file = create_temp_file(suffix=".wav")
            extract(temp_file, extracted_file)

        result = process(temp_file or extracted_file)
        return result

    finally:
        # Cleanup all temp files
        if temp_file.exists():
            temp_file.unlink()
        if extracted_file is not None and extracted_file.exists():
            extracted_file.unlink()
```

### Logging Pattern

```python
# Use Rich console utilities, not print()
from gtranscriber.utils.logger import (
    print_info,
    print_success,
    print_error,
    print_warning,
)

# Use Python logging for detailed logs
import logging
logger = logging.getLogger(__name__)

# In processing code
logger.info("Starting process")
logger.debug("Detailed debug info")
logger.warning("Non-fatal warning")
logger.error("Error occurred")
logger.exception("Error with stack trace")

# In CLI code
print_info("Starting batch processing")
print_success("Completed successfully!")
print_error("Processing failed")
```

---

## Applying Patterns to Phase 2

### QA Generation Pipeline

Following the batch processing pattern, here's how to structure QA generation:

#### 1. Configuration (`config.py`)

```python
class QAConfig(BaseSettings):
    """Already implemented in Phase 1 ✅"""
    model_config = SettingsConfigDict(env_prefix="GTRANSCRIBER_QA_")

    provider: str = "ollama"
    model_id: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    base_url: str | None = None
    questions_per_document: int = 10
    strategies: list[str] = ["factual", "conceptual"]
    temperature: float = 0.7
    output_dir: Path = Path("qa_dataset")
```

#### 2. Task Definition (`qa_batch.py`)

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class QAGenerationTask:
    """Task for QA generation."""

    transcription_file: Path
    gdrive_id: str
    filename: str
    output_file: Path
```

#### 3. Worker Function (`qa_batch.py`)

```python
# Global QA generator (one per process)
_worker_qa_generator: QAGenerator | None = None

def _init_qa_worker(
    provider: str,
    model_id: str,
    config: QAConfig,
) -> None:
    """Initialize worker with QA generator."""
    global _worker_qa_generator

    from gtranscriber.core.llm_client import LLMProvider, LLMClient
    from gtranscriber.core.qa_generator import QAGenerator

    # Create LLM client
    llm_client = LLMClient(
        provider=LLMProvider(provider),
        model_id=model_id,
        base_url=config.base_url or config.ollama_url,
    )

    # Create QA generator
    _worker_qa_generator = QAGenerator(llm_client, config)
    logger.info(f"QA worker initialized with {provider}/{model_id}")

def generate_qa_for_transcription(
    task: QAGenerationTask,
    config: QAConfig,
) -> tuple[str, bool, str]:
    """Generate QA pairs for a single transcription (worker function).

    Returns:
        Tuple of (gdrive_id, success, message).
    """
    global _worker_qa_generator

    try:
        # For sequential, initialize on first use
        if _worker_qa_generator is None:
            _init_qa_worker(config.provider, config.model_id, config)

        # Load transcription
        with open(task.transcription_file) as f:
            enriched = EnrichedRecord(**json.load(f))

        # Generate QA pairs
        qa_record = _worker_qa_generator.generate_qa_pairs(enriched)

        # Save result
        task.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(task.output_file, 'w') as f:
            json.dump(qa_record.model_dump(), f, indent=2)

        logger.info(f"Generated {len(qa_record.qa_pairs)} QA pairs for {task.filename}")
        return task.gdrive_id, True, "Success"

    except Exception as e:
        logger.exception(f"QA generation failed: {task.filename}")
        return task.gdrive_id, False, str(e)
```

#### 4. Batch Orchestrator (`qa_batch.py`)

```python
def run_batch_qa_generation(
    input_dir: Path,
    output_dir: Path,
    config: QAConfig,
    num_workers: int = 2,
) -> None:
    """Run batch QA generation with checkpointing.

    Args:
        input_dir: Directory containing EnrichedRecord JSONs
        output_dir: Directory for QARecord JSONs
        config: QA generation configuration
        num_workers: Number of parallel workers
    """
    # 1. Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "qa_checkpoint.json"
    checkpoint = CheckpointManager(checkpoint_file)

    # 2. Load tasks
    all_tasks = []
    for json_file in input_dir.glob("*_transcription.json"):
        with open(json_file) as f:
            data = json.load(f)
            gdrive_id = data.get("gdrive_id", "unknown")

            task = QAGenerationTask(
                transcription_file=json_file,
                gdrive_id=gdrive_id,
                filename=data.get("name", json_file.name),
                output_file=output_dir / f"{gdrive_id}_qa.json",
            )
            all_tasks.append(task)

    # 3. Filter completed
    remaining_tasks = [
        t for t in all_tasks
        if not checkpoint.is_completed(t.gdrive_id)
    ]
    checkpoint.set_total_files(len(all_tasks))

    logger.info(f"Total: {len(all_tasks)}, Remaining: {len(remaining_tasks)}")

    if not remaining_tasks:
        logger.info("All files already processed!")
        return

    # 4. Process
    num_workers = min(num_workers, len(remaining_tasks))

    if num_workers == 1:
        # Sequential
        for task in remaining_tasks:
            gdrive_id, success, message = generate_qa_for_transcription(task, config)
            if success:
                checkpoint.mark_completed(gdrive_id)
            else:
                checkpoint.mark_failed(gdrive_id, message)
    else:
        # Parallel
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_qa_worker,
            initargs=(config.provider, config.model_id, config),
        ) as executor:
            # Use batched submission pattern
            batch_size = max(num_workers * 2, 10)
            task_iter = iter(remaining_tasks)
            pending = {}

            # Initial batch
            for _ in range(min(batch_size, len(remaining_tasks))):
                task = next(task_iter)
                future = executor.submit(generate_qa_for_transcription, task, config)
                pending[future] = task

            # Process completions
            while pending:
                done = next(as_completed(pending))
                task = pending.pop(done)

                gdrive_id, success, message = done.result()

                if success:
                    checkpoint.mark_completed(gdrive_id)
                    logger.info(f"✓ {task.filename}")
                else:
                    checkpoint.mark_failed(gdrive_id, message)
                    logger.error(f"✗ {task.filename}: {message}")

                # Submit next
                try:
                    next_task = next(task_iter)
                    next_future = executor.submit(
                        generate_qa_for_transcription, next_task, config
                    )
                    pending[next_future] = next_task
                except StopIteration:
                    pass

    # 5. Summary
    completed, total = checkpoint.get_progress()
    failed = len(checkpoint.state.failed_files)
    logger.info(f"Completed: {completed}/{total}, Failed: {failed}")
```

#### 5. CLI Command (`main.py`)

```python
@app.command()
def generate_qa(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing transcription JSON files",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o",
            help="Output directory for QA datasets",
        ),
    ] = Path("qa_dataset"),
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            help="LLM provider: openai, ollama, custom",
        ),
    ] = "ollama",
    model_id: Annotated[
        str,
        typer.Option(
            "--model-id", "-m",
            help="Model ID for QA generation",
        ),
    ] = "llama3.1:8b",
    workers: Annotated[
        int,
        typer.Option(
            "--workers", "-w",
            help="Number of parallel workers",
        ),
    ] = 2,
    questions: Annotated[
        int,
        typer.Option(
            "--questions",
            help="Number of QA pairs per document",
        ),
    ] = 10,
    strategy: Annotated[
        list[str],
        typer.Option(
            "--strategy",
            help="Question strategies (can specify multiple)",
        ),
    ] = ["factual", "conceptual"],
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            help="LLM temperature",
        ),
    ] = 0.7,
) -> None:
    """Generate synthetic QA pairs from transcriptions.

    Processes all transcription JSON files in the input directory and
    generates question-answer pairs using the specified LLM provider.

    Progress is automatically checkpointed, allowing interrupted jobs
    to resume from the last completed file.
    """
    from gtranscriber.config import QAConfig
    from gtranscriber.core.qa_batch import run_batch_qa_generation

    # Validate
    if workers < 1:
        print_error("Workers must be at least 1")
        raise typer.Exit(code=1)

    if questions < 1 or questions > 50:
        print_error("Questions must be between 1 and 50")
        raise typer.Exit(code=1)

    # Create config
    qa_config = QAConfig(
        provider=provider,
        model_id=model_id,
        questions_per_document=questions,
        strategies=strategy,
        temperature=temperature,
        output_dir=output_dir,
    )

    # Display config
    console.print("\n[bold]QA Generation Configuration[/bold]\n")
    console.print(f"[cyan]Input Directory:[/cyan] {input_dir}")
    console.print(f"[cyan]Output Directory:[/cyan] {output_dir}")
    console.print(f"[cyan]Provider:[/cyan] {provider}")
    console.print(f"[cyan]Model:[/cyan] {model_id}")
    console.print(f"[cyan]Workers:[/cyan] {workers}")
    console.print(f"[cyan]Questions per doc:[/cyan] {questions}")
    console.print(f"[cyan]Strategies:[/cyan] {', '.join(strategy)}")
    console.print()

    try:
        run_batch_qa_generation(input_dir, output_dir, qa_config, workers)
        print_success("QA generation completed!")

    except Exception as e:
        print_error(f"QA generation failed: {e}")
        raise typer.Exit(code=1) from e
```

---

## Summary: Phase 2 Implementation Checklist

### Files to Create

- [ ] `src/gtranscriber/core/qa_generator.py` - QA generation logic
- [ ] `src/gtranscriber/core/qa_batch.py` - Batch orchestrator
- [ ] Extend `src/gtranscriber/main.py` - Add `generate_qa` command

### Patterns to Follow

- [ ] Use `@dataclass` for task and config objects
- [ ] Implement global worker initialization pattern
- [ ] Use `CheckpointManager` for progress tracking
- [ ] Return `(id, success, message)` from worker functions
- [ ] Use batched future submission in parallel mode
- [ ] Clean up resources in `finally` blocks
- [ ] Use `print_info/success/error` for user output
- [ ] Use `logger.info/debug/error` for detailed logs
- [ ] Validate inputs before processing
- [ ] Display configuration before running

### Architecture Consistency

```python
# Each pipeline follows this structure:

1. Config class (Pydantic BaseSettings)
   └─ Environment variables with GTRANSCRIBER_{PIPELINE}_ prefix

2. Task dataclass
   └─ Simple container for task parameters

3. Worker initialization
   └─ Global engine/generator, initialized once per process

4. Worker function
   └─ Signature: (task, config) -> (id, success, message)
   └─ Never raises, always returns triple

5. Batch orchestrator
   └─ Load tasks, filter completed, run with checkpoint

6. CLI command
   └─ Validate, create config, display, run, handle errors
```

---

**Next Steps**: Implement Phase 2 following these patterns for consistency and maintainability.

**References**:
- [Implementation Plan - Phase 2](IMPLEMENTATION_PLAN.md#phase-2-qa-generation-week-2)
- [Data Schemas](implementation/DATA_SCHEMAS.md)
- [CLI Reference](implementation/CLI_REFERENCE.md)

---

**Document Version**: 1.0
**Last Updated**: 2026-01-29
**Status**: ✅ Ready for Phase 2 Implementation
