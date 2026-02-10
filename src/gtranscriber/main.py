"""G-Transcriber CLI interface.

Main entry point for the G-Transcriber application using Typer for CLI
and Rich for visual feedback.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from gtranscriber import __version__
from gtranscriber.config import ResultsConfig, TranscriberConfig
from gtranscriber.core.engine import WhisperEngine
from gtranscriber.core.hardware import get_device_and_dtype
from gtranscriber.core.io import (
    create_temp_file,
    get_mime_type,
    get_output_filename,
    save_enriched_record,
)
from gtranscriber.core.transcription_validator import validate_enriched_record
from gtranscriber.schemas import EnrichedRecord, InputRecord, TranscriptionSegment
from gtranscriber.utils.console import console
from gtranscriber.utils.logger import (
    print_error,
    print_info,
    print_success,
    print_warning,
    setup_logging,
)
from gtranscriber.utils.ui import (
    create_progress,
    display_config_table,
    display_result_panel,
)

if TYPE_CHECKING:
    from gtranscriber.core.engine import TranscriptionResult

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize Typer app
app = typer.Typer(
    name="gtranscriber",
    help="Automated transcription system for media files using Whisper ASR.",
    add_completion=False,
    rich_markup_mode="rich",
)

# Load configuration from environment variables
# This is loaded once at module import time and used as defaults for CLI parameters.
# CLI arguments will override these defaults, but if not specified, values from
# environment variables (GTRANSCRIBER_*) or .env file will be used.
_config = TranscriberConfig()
_results_config = ResultsConfig()
logger.debug(f"Loaded configuration:\n{json.dumps(_config.model_dump_json(), indent=2)}")
# Default paths from config - these are used as CLI parameter defaults
DEFAULT_CREDENTIALS_PATH = Path(_config.credentials)
DEFAULT_TOKEN_PATH = Path(_config.token)


def _ensure_float(value: Any, default: float) -> float:
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
    """Create TranscriptionSegment list from TranscriptionResult.

    Args:
        result: Transcription result containing segments.

    Returns:
        List of TranscriptionSegment objects or None if no segments.
    """
    if result.segments is None:
        return None

    if not result.segments:
        return []

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


def _safe_int_conversion(value: str | None, default: int | None = None) -> int | None:
    """Safely convert a string value to integer.

    Args:
        value: String value to convert.
        default: Default value if conversion fails.

    Returns:
        Integer value or default.
    """
    if value is None:
        return default
    try:
        # Try converting to float first to handle float strings like "42.7"
        return int(float(value))
    except (ValueError, TypeError):
        return default


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold]G-Transcriber[/bold] version {__version__}")
        raise typer.Exit(code=0)


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """G-Transcriber: Automated transcription for Google Drive media files."""
    setup_logging()


@app.command()
def transcribe(
    file_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the audio/video file to transcribe.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model_id: Annotated[
        str,
        typer.Option(
            "--model-id",
            "-m",
            help="Hugging Face model ID for transcription. "
            "Can be set via GTRANSCRIBER_MODEL_ID env var.",
        ),
    ] = _config.model_id,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output file path for transcription JSON.",
        ),
    ] = None,
    quantize: Annotated[
        bool,
        typer.Option(
            "--quantize",
            "-q",
            help="Enable 8-bit quantization to reduce VRAM usage. "
            "Can be set via GTRANSCRIBER_QUANTIZE env var.",
        ),
    ] = _config.quantize,
    cpu: Annotated[
        bool,
        typer.Option(
            "--cpu",
            help="Force CPU execution (disables CUDA/MPS even if available). "
            "Can be set via GTRANSCRIBER_FORCE_CPU env var.",
        ),
    ] = _config.force_cpu,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language code for transcription (e.g., 'pt' for Portuguese). "
            "If not specified, the language will be auto-detected. "
            "Can be set via GTRANSCRIBER_LANGUAGE env var.",
        ),
    ] = _config.language,
) -> None:
    """Transcribe a local audio or video file.

    The transcription result will be saved as a JSON file containing both
    the transcription text and metadata about the processing.
    """
    # Get hardware configuration
    hw_config = get_device_and_dtype(force_cpu=cpu, quantize=quantize)

    # Display configuration
    display_config_table(
        model_id=model_id,
        device=hw_config.device,
        quantize=quantize,
        source=str(file_path),
    )

    print_info(f"Starting transcription of [bold]{file_path.name}[/bold]")

    try:
        # Initialize engine
        with create_progress("Initializing model") as progress:
            task = progress.add_task("Loading model...", total=100)
            engine = WhisperEngine(
                model_id=model_id,
                force_cpu=cpu,
                quantize=quantize,
                language=language,
            )
            progress.update(task, completed=100)

        # Perform transcription
        with create_progress("Transcribing") as progress:
            task = progress.add_task("Processing audio...", total=None)
            result = engine.transcribe(file_path)
            progress.update(task, completed=100)

        # Create enriched record
        segments = _create_segments_from_result(result)

        # Create a minimal input record for local files
        enriched = EnrichedRecord(
            gdrive_id="local",
            name=file_path.name,
            mimeType=get_mime_type(file_path),
            parents=[],
            webContentLink="local://",
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
            print_warning(
                f"Quality issues detected: {enriched.transcription_quality.issues_detected}"
            )

        # Determine output path
        if output is None:
            output = file_path.parent / get_output_filename(file_path.name)

        # Save result
        save_enriched_record(enriched, output)

        # Display result
        display_result_panel(enriched)
        print_success(f"Transcription saved to [bold]{output}[/bold]")

    except Exception as e:
        print_error(f"Transcription failed: {e}")
        raise typer.Exit(code=1) from e


@app.command()
def drive_transcribe(
    file_id: Annotated[
        str,
        typer.Argument(
            help="Google Drive file ID to transcribe.",
        ),
    ],
    model_id: Annotated[
        str,
        typer.Option(
            "--model-id",
            "-m",
            help="Hugging Face model ID for transcription. "
            "Can be set via GTRANSCRIBER_MODEL_ID env var.",
        ),
    ] = _config.model_id,
    credentials: Annotated[
        Path,
        typer.Option(
            "--credentials",
            "-c",
            help="Path to Google OAuth2 credentials file. "
            "Can be set via GTRANSCRIBER_CREDENTIALS env var.",
        ),
    ] = DEFAULT_CREDENTIALS_PATH,
    token: Annotated[
        Path,
        typer.Option(
            "--token",
            "-t",
            help="Path to Google OAuth2 token file. Can be set via GTRANSCRIBER_TOKEN env var.",
        ),
    ] = DEFAULT_TOKEN_PATH,
    quantize: Annotated[
        bool,
        typer.Option(
            "--quantize",
            "-q",
            help="Enable 8-bit quantization to reduce VRAM usage. "
            "Can be set via GTRANSCRIBER_QUANTIZE env var.",
        ),
    ] = _config.quantize,
    cpu: Annotated[
        bool,
        typer.Option(
            "--cpu",
            help="Force CPU execution. Can be set via GTRANSCRIBER_FORCE_CPU env var.",
        ),
    ] = _config.force_cpu,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language code for transcription (e.g., 'pt' for Portuguese). "
            "If not specified, the language will be auto-detected. "
            "Can be set via GTRANSCRIBER_LANGUAGE env var.",
        ),
    ] = _config.language,
) -> None:
    """Transcribe a file from Google Drive.

    Downloads the file, transcribes it, and uploads the result to the same
    Drive folder as the original file.
    """
    from gtranscriber.core.drive import DriveClient

    # Get hardware configuration
    hw_config = get_device_and_dtype(force_cpu=cpu, quantize=quantize)

    print_info(f"Fetching file metadata for ID: [bold]{file_id}[/bold]")

    try:
        # Initialize Drive client
        drive_client = DriveClient(
            credentials_file=str(credentials),
            token_file=str(token),
        )

        # Get file metadata
        metadata = drive_client.get_file_metadata(file_id)

        # Validate input
        input_record = InputRecord(
            gdrive_id=metadata["id"],
            name=metadata["name"],
            mimeType=metadata["mimeType"],
            parents=metadata.get("parents", []),
            webContentLink=metadata.get("webContentLink", ""),
            size_bytes=_safe_int_conversion(metadata.get("size")),
        )

        display_config_table(
            model_id=model_id,
            device=hw_config.device,
            quantize=quantize,
            source=input_record.name,
        )

        # Download file
        suffix = Path(input_record.name).suffix
        temp_file = create_temp_file(suffix=suffix)
        local_output = None

        try:
            with create_progress("Downloading") as progress:
                task = progress.add_task(f"Downloading {input_record.name}...", total=100)
                drive_client.download_file(file_id, temp_file, progress, task)

            print_success(f"Downloaded [bold]{input_record.name}[/bold]")

            # Initialize engine and transcribe
            with create_progress("Initializing model") as progress:
                task = progress.add_task("Loading model...", total=100)
                engine = WhisperEngine(
                    model_id=model_id,
                    force_cpu=cpu,
                    quantize=quantize,
                    language=language,
                )
                progress.update(task, completed=100)

            with create_progress("Transcribing") as progress:
                task = progress.add_task("Processing audio...", total=None)
                result = engine.transcribe(temp_file)
                progress.update(task, completed=100)

            # Create enriched record
            segments = _create_segments_from_result(result)

            enriched = EnrichedRecord(
                gdrive_id=input_record.gdrive_id,
                name=input_record.name,
                mimeType=input_record.mimeType,
                parents=input_record.parents,
                webContentLink=input_record.web_content_link,
                size_bytes=input_record.size_bytes,
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
                print_warning(
                    f"Quality issues detected: {enriched.transcription_quality.issues_detected}"
                )

            # Save locally first
            output_filename = get_output_filename(input_record.name)
            local_output = temp_file.parent / output_filename
            save_enriched_record(enriched, local_output)

            # Upload to Drive (same folder as original)
            if input_record.parents:
                with create_progress("Uploading") as progress:
                    task = progress.add_task("Uploading result...", total=100)
                    upload_result = drive_client.upload_file(
                        local_output,
                        input_record.parents[0],
                        progress=progress,
                        task_id=task,
                    )
                print_success(
                    f"Uploaded transcription to Drive: [bold]{upload_result.get('name')}[/bold]"
                )

            # Update original file properties
            drive_client.update_file_properties(
                file_id,
                {
                    "x-transcription-status": "completed",
                    "x-model-id": model_id,
                    "x-transcription-date": datetime.now().isoformat(),
                },
            )

            # Display result
            display_result_panel(enriched)

        finally:
            # Cleanup temporary files with logging
            if temp_file.exists():
                temp_file.unlink()
                logger.debug(f"Deleted temporary file: {temp_file}")
            if local_output and local_output.exists():
                local_output.unlink()
                logger.debug(f"Deleted local output file: {local_output}")

    except Exception as e:
        print_error(f"Transcription failed: {e}")
        raise typer.Exit(code=1) from e


@app.command()
def batch_transcribe(
    catalog: Annotated[
        Path,
        typer.Argument(
            help="Path to catalog CSV file with Google Drive file metadata.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for transcription JSON files. "
            "Can be set via GTRANSCRIBER_RESULTS_DIR env var.",
        ),
    ] = Path(_config.results_dir),
    model_id: Annotated[
        str,
        typer.Option(
            "--model-id",
            "-m",
            help="Hugging Face model ID for transcription. "
            "Can be set via GTRANSCRIBER_MODEL_ID env var.",
        ),
    ] = _config.model_id,
    credentials: Annotated[
        Path,
        typer.Option(
            "--credentials",
            "-c",
            help="Path to Google OAuth2 credentials file. "
            "Can be set via GTRANSCRIBER_CREDENTIALS env var.",
        ),
    ] = DEFAULT_CREDENTIALS_PATH,
    token: Annotated[
        Path,
        typer.Option(
            "--token",
            "-t",
            help="Path to Google OAuth2 token file. Can be set via GTRANSCRIBER_TOKEN env var.",
        ),
    ] = DEFAULT_TOKEN_PATH,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help="Number of parallel workers (each loads its own model instance). "
            "Can be set via GTRANSCRIBER_WORKERS env var.",
        ),
    ] = _config.workers,
    checkpoint_file: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            help="Path to checkpoint file for resuming interrupted jobs.",
        ),
    ] = Path("results/checkpoint.json"),
    quantize: Annotated[
        bool,
        typer.Option(
            "--quantize",
            "-q",
            help="Enable 8-bit quantization to reduce VRAM usage. "
            "Can be set via GTRANSCRIBER_QUANTIZE env var.",
        ),
    ] = _config.quantize,
    cpu: Annotated[
        bool,
        typer.Option(
            "--cpu",
            help="Force CPU execution. Can be set via GTRANSCRIBER_FORCE_CPU env var.",
        ),
    ] = _config.force_cpu,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language code for transcription (e.g., 'pt' for Portuguese). "
            "If not specified, the language will be auto-detected. "
            "Can be set via GTRANSCRIBER_LANGUAGE env var.",
        ),
    ] = _config.language,
) -> None:
    """Batch transcribe audio/video files from a catalog.

    Processes all audio and video files listed in the catalog CSV with parallel
    processing support and automatic checkpoint/resume capability. Each worker
    loads its own model instance for true parallel processing.

    The catalog CSV must contain columns: gdrive_id, name, mime_type, size_bytes,
    parents, web_content_link, and optionally duration_milliseconds.

    Transcription results are saved as JSON files in the output directory with
    all metadata including media duration. Progress is automatically checkpointed,
    allowing interrupted jobs to resume from the last completed file.
    """
    from gtranscriber.core.batch import BatchConfig, run_batch_transcription

    # Validate inputs
    if not credentials.exists():
        print_error(f"Credentials file not found: {credentials}")
        raise typer.Exit(code=1)

    if not token.exists():
        print_error(f"Token file not found: {token}")
        raise typer.Exit(code=1)

    if workers < 1:
        print_error("Number of workers must be at least 1")
        raise typer.Exit(code=1)

    # Create config
    config = BatchConfig(
        catalog_file=catalog,
        output_dir=output_dir,
        checkpoint_file=checkpoint_file,
        credentials_file=credentials,
        token_file=token,
        model_id=model_id,
        num_workers=workers,
        force_cpu=cpu,
        quantize=quantize,
        language=language,
    )

    # Display configuration
    console.print("\n[bold]Batch Transcription Configuration[/bold]\n")
    console.print(f"[cyan]Catalog:[/cyan] {catalog}")
    console.print(f"[cyan]Output Directory:[/cyan] {output_dir}")
    console.print(f"[cyan]Model ID:[/cyan] {model_id}")
    console.print(f"[cyan]Workers:[/cyan] {workers}")
    console.print(f"[cyan]Checkpoint:[/cyan] {checkpoint_file}")
    console.print(f"[cyan]Quantize:[/cyan] {quantize}")
    console.print(f"[cyan]Force CPU:[/cyan] {cpu}")
    console.print(f"[cyan]Language:[/cyan] {language or 'auto-detect'}")
    console.print()

    try:
        run_batch_transcription(config)
        print_success("Batch transcription completed!")

    except Exception as e:
        print_error(f"Batch transcription failed: {e}")
        raise typer.Exit(code=1) from e


@app.command()
def refresh_auth(
    credentials: Annotated[
        Path,
        typer.Option(
            "--credentials",
            "-c",
            help="Path to Google OAuth2 credentials file.",
        ),
    ] = Path("credentials.json"),
    token: Annotated[
        Path,
        typer.Option(
            "--token",
            "-t",
            help="Path to token file to refresh.",
        ),
    ] = Path("token.json"),
) -> None:
    """Fully refresh Google OAuth2 authentication token.

    This command deletes the existing token file and initiates a fresh OAuth2
    authorization flow. Use this when you need to:

    - Re-authorize with different Google account
    - Fix authentication issues or permission problems
    - Update token after revoking access in Google Account settings
    """
    from gtranscriber.core.drive import DriveClient

    # Check if credentials file exists
    if not credentials.exists():
        print_error(f"Credentials file not found: {credentials}")
        raise typer.Exit(code=1)

    # Delete existing token if it exists
    if token.exists():
        print_info(f"Removing existing token: [bold]{token}[/bold]")
        token.unlink()

    print_info("Starting OAuth2 authorization flow...")
    print_info("A browser window will open for Google authentication.")

    try:
        # Initialize DriveClient which triggers fresh authentication
        client = DriveClient(
            credentials_file=str(credentials),
            token_file=str(token),
        )
        # Access service property to trigger authentication
        _ = client.service

        print_success("Authentication successful!")
        print_success(f"Token saved to: [bold]{token}[/bold]")

    except Exception as e:
        print_error(f"Authentication failed: {e}")
        raise typer.Exit(code=1) from e


@app.command()
def generate_qa(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing transcription JSON files.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for QA dataset JSON files. "
            "Can be set via GTRANSCRIBER_QA_OUTPUT_DIR env var.",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="LLM provider: openai, ollama, custom. "
            "Can be set via GTRANSCRIBER_QA_PROVIDER env var.",
        ),
    ] = None,
    model_id: Annotated[
        str | None,
        typer.Option(
            "--model-id",
            "-m",
            help="Model ID for QA generation (e.g., llama3.1:8b, gpt-4). "
            "Can be set via GTRANSCRIBER_QA_MODEL_ID env var.",
        ),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            help="Number of parallel workers. Can be set via GTRANSCRIBER_QA_WORKERS env var.",
        ),
    ] = None,
    questions: Annotated[
        int | None,
        typer.Option(
            "--questions",
            help="Number of QA pairs to generate per document (1-50). "
            "Can be set via GTRANSCRIBER_QA_QUESTIONS_PER_DOCUMENT env var.",
        ),
    ] = None,
    strategy: Annotated[
        list[str] | None,
        typer.Option(
            "--strategy",
            help="Question generation strategies (factual, conceptual, temporal, entity). "
            "Can specify multiple times. "
            "Can be set via GTRANSCRIBER_QA_STRATEGIES env var.",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            "--temperature",
            help="LLM temperature for generation (0.0-2.0). "
            "Can be set via GTRANSCRIBER_QA_TEMPERATURE env var.",
        ),
    ] = None,
    ollama_url: Annotated[
        str | None,
        typer.Option(
            "--ollama-url",
            help="Ollama API base URL. Can be set via GTRANSCRIBER_QA_OLLAMA_URL env var.",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="Custom base URL for OpenAI-compatible endpoints. "
            "Can be set via GTRANSCRIBER_QA_BASE_URL env var.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language for QA prompts: 'en' (English) or 'pt' (Portuguese). "
            "Can be set via GTRANSCRIBER_QA_LANGUAGE env var.",
        ),
    ] = None,
) -> None:
    """Generate synthetic QA pairs from transcriptions.

    Processes all transcription JSON files in the input directory and generates
    question-answer pairs using the specified LLM provider. Supports multiple
    question generation strategies (factual, conceptual, temporal, entity-focused)
    to create diverse evaluation datasets.

    Progress is automatically checkpointed, allowing interrupted jobs to resume
    from the last completed file. QA pairs are saved as JSON files in the output
    directory.

    Examples:
        # Generate QA pairs using Ollama
        gtranscriber generate-qa results/ -o qa_dataset/

        # Use OpenAI with specific model
        gtranscriber generate-qa results/ --provider openai --model-id gpt-4

        # Generate more pairs with multiple strategies
        gtranscriber generate-qa results/ --questions 20 \\
            --strategy factual --strategy conceptual --strategy temporal

        # Use multiple workers for faster processing
        gtranscriber generate-qa results/ --workers 4
    """
    from gtranscriber.config import QAConfig
    from gtranscriber.core.qa_batch import run_batch_qa_generation

    # Load config first to get defaults from environment variables
    base_config = QAConfig()

    # Override with CLI args if provided (None means use config default)
    if provider is not None:
        base_config.provider = provider
    if model_id is not None:
        base_config.model_id = model_id
    if ollama_url is not None:
        base_config.ollama_url = ollama_url
    if base_url is not None:
        base_config.base_url = base_url
    if questions is not None:
        base_config.questions_per_document = questions
    if strategy is not None:
        base_config.strategies = strategy
    if temperature is not None:
        base_config.temperature = temperature
    if output_dir is not None:
        base_config.output_dir = output_dir
    if workers is not None:
        base_config.workers = workers
    if language is not None:
        base_config.language = language

    # Now use the resolved config
    qa_config = base_config

    # Validate resolved values
    if qa_config.workers < 1:
        print_error("Number of workers must be at least 1")
        raise typer.Exit(code=1)

    if qa_config.questions_per_document < 1 or qa_config.questions_per_document > 50:
        print_error("Number of questions must be between 1 and 50")
        raise typer.Exit(code=1)

    if qa_config.temperature < 0.0 or qa_config.temperature > 2.0:
        print_error("Temperature must be between 0.0 and 2.0")
        raise typer.Exit(code=1)

    # Validate strategies
    valid_strategies = {"factual", "conceptual", "temporal", "entity"}
    for s in qa_config.strategies:
        if s not in valid_strategies:
            print_error(f"Invalid strategy: {s!r}. Must be one of {sorted(valid_strategies)}")
            raise typer.Exit(code=1)

    # Validate language
    valid_languages = {"en", "pt"}
    if qa_config.language not in valid_languages:
        print_error(
            f"Invalid language: {qa_config.language!r}. Must be one of {sorted(valid_languages)}"
        )
        raise typer.Exit(code=1)

    # Display configuration
    console.print("\n[bold]QA Generation Configuration[/bold]\n")
    console.print(f"[cyan]Input Directory:[/cyan] {input_dir}")
    console.print(f"[cyan]Output Directory:[/cyan] {qa_config.output_dir}")
    console.print(f"[cyan]Provider:[/cyan] {qa_config.provider}")
    console.print(f"[cyan]Model:[/cyan] {qa_config.model_id}")
    console.print(f"[cyan]Workers:[/cyan] {qa_config.workers}")
    console.print(f"[cyan]Questions per document:[/cyan] {qa_config.questions_per_document}")
    console.print(f"[cyan]Strategies:[/cyan] {', '.join(qa_config.strategies)}")
    console.print(f"[cyan]Language:[/cyan] {qa_config.language}")
    console.print(f"[cyan]Temperature:[/cyan] {qa_config.temperature}")
    if qa_config.provider == "ollama":
        console.print(f"[cyan]Ollama URL:[/cyan] {qa_config.ollama_url}")
    if qa_config.base_url:
        console.print(f"[cyan]Base URL:[/cyan] {qa_config.base_url}")
    console.print()

    try:
        run_batch_qa_generation(input_dir, qa_config.output_dir, qa_config, qa_config.workers)
        print_success("QA generation completed!")

    except Exception as e:
        print_error(f"QA generation failed: {e}")
        raise typer.Exit(code=1) from e


@app.command()
def generate_cep_qa(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing transcription JSON files.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for CEP QA dataset JSON files.",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="LLM provider: openai, ollama, custom.",
        ),
    ] = None,
    model_id: Annotated[
        str | None,
        typer.Option(
            "--model-id",
            "-m",
            help="Model ID for QA generation (e.g., llama3.1:8b, gpt-4).",
        ),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            help="Number of parallel workers.",
        ),
    ] = None,
    questions: Annotated[
        int | None,
        typer.Option(
            "--questions",
            help="Number of QA pairs to generate per document (1-50).",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            "--temperature",
            help="LLM temperature for generation (0.0-2.0).",
        ),
    ] = None,
    ollama_url: Annotated[
        str | None,
        typer.Option(
            "--ollama-url",
            help="Ollama API base URL.",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="Custom base URL for OpenAI-compatible endpoints.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language for prompts: 'pt' (Portuguese) or 'en' (English). Default: pt",
        ),
    ] = None,
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Enable LLM-as-a-Judge validation. Default: enabled",
        ),
    ] = True,
    validator_model: Annotated[
        str | None,
        typer.Option(
            "--validator-model",
            help="Model ID for LLM-as-a-Judge validation.",
        ),
    ] = None,
    bloom_dist: Annotated[
        str | None,
        typer.Option(
            "--bloom-dist",
            help="Bloom level distribution as 'level:weight,...' "
            "(e.g., 'remember:0.2,understand:0.3,analyze:0.3,evaluate:0.2').",
        ),
    ] = None,
    export_jsonl: Annotated[
        bool,
        typer.Option(
            "--jsonl/--no-jsonl",
            help="Also export QA pairs to JSONL format for KGQA training.",
        ),
    ] = False,
) -> None:
    """Generate CEP (cognitive scaffolding) QA pairs from transcriptions.

    Uses the Cognitive Elicitation Pipeline (CEP) with:
    - Module I: Bloom Scaffolding (question generation by cognitive level)
    - Module II: Reasoning & Grounding (reasoning traces and multi-hop detection)
    - Module III: LLM-as-a-Judge Validation (optional quality evaluation)

    Questions are distributed across Bloom taxonomy levels (remember, understand,
    analyze, evaluate) to create cognitively scaffolded QA datasets.

    Examples:
        # Basic CEP generation (default: Portuguese, validation enabled)
        gtranscriber generate-cep-qa results/ -o cep_dataset/

        # Disable validation for faster processing
        gtranscriber generate-cep-qa results/ --no-validate

        # Use custom validator model
        gtranscriber generate-cep-qa results/ --validator-model gpt-4

        # Adjust Bloom level distribution
        gtranscriber generate-cep-qa results/ \\
            --bloom-dist "remember:0.1,understand:0.3,analyze:0.4,evaluate:0.2"

        # Export to JSONL for KGQA training
        gtranscriber generate-cep-qa results/ --jsonl
    """
    from gtranscriber.config import CEPConfig, QAConfig
    from gtranscriber.core.qa_batch import run_batch_cep_generation

    # Load configs with defaults from environment variables
    qa_config = QAConfig()
    cep_config = CEPConfig()

    # Build QA config overrides dict for CLI args
    # Using model_validate ensures validators run on CLI-provided values
    qa_overrides: dict[str, Any] = {}
    if provider is not None:
        qa_overrides["provider"] = provider
    if model_id is not None:
        qa_overrides["model_id"] = model_id
    if ollama_url is not None:
        qa_overrides["ollama_url"] = ollama_url
    if base_url is not None:
        qa_overrides["base_url"] = base_url
    if questions is not None:
        qa_overrides["questions_per_document"] = questions
    if temperature is not None:
        qa_overrides["temperature"] = temperature
    if output_dir is not None:
        qa_overrides["output_dir"] = output_dir
    if workers is not None:
        qa_overrides["workers"] = workers

    if qa_overrides:
        qa_config = QAConfig.model_validate({**qa_config.model_dump(), **qa_overrides})

    # Build CEP config overrides dict for CLI args
    cep_overrides: dict[str, Any] = {}
    if language is not None:
        cep_overrides["language"] = language
    if not validate:
        cep_overrides["enable_validation"] = False
    if validator_model is not None:
        cep_overrides["validator_model_id"] = validator_model

    # Parse Bloom distribution if provided
    if bloom_dist is not None:
        try:
            dist_dict = {}
            for item in bloom_dist.split(","):
                level, weight = item.strip().split(":")
                dist_dict[level.strip()] = float(weight.strip())
            cep_overrides["bloom_distribution"] = dist_dict
            cep_overrides["bloom_levels"] = list(dist_dict.keys())
        except ValueError as e:
            print_error(f"Invalid bloom-dist format: {e}")
            print_error("Expected format: 'level:weight,level:weight,...'")
            raise typer.Exit(code=1) from e

    if cep_overrides:
        try:
            cep_config = CEPConfig.model_validate({**cep_config.model_dump(), **cep_overrides})
        except ValueError as e:
            print_error(f"Invalid CEP configuration: {e}")
            raise typer.Exit(code=1) from e

    # Validate configs
    if qa_config.workers < 1:
        print_error("Number of workers must be at least 1")
        raise typer.Exit(code=1)

    if qa_config.questions_per_document < 1 or qa_config.questions_per_document > 50:
        print_error("Number of questions must be between 1 and 50")
        raise typer.Exit(code=1)

    valid_languages = {"en", "pt"}
    if cep_config.language not in valid_languages:
        print_error(
            f"Invalid language: {cep_config.language!r}. Must be one of {sorted(valid_languages)}"
        )
        raise typer.Exit(code=1)

    # Display configuration
    console.print("\n[bold]CEP QA Generation Configuration[/bold]\n")
    console.print(f"[cyan]Input Directory:[/cyan] {input_dir}")
    console.print(f"[cyan]Output Directory:[/cyan] {qa_config.output_dir}")
    console.print(f"[cyan]Provider:[/cyan] {qa_config.provider}")
    console.print(f"[cyan]Model:[/cyan] {qa_config.model_id}")
    console.print(f"[cyan]Workers:[/cyan] {qa_config.workers}")
    console.print(f"[cyan]Questions per document:[/cyan] {qa_config.questions_per_document}")
    console.print(f"[cyan]Language:[/cyan] {cep_config.language}")
    console.print(f"[cyan]Bloom Levels:[/cyan] {', '.join(cep_config.bloom_levels)}")
    console.print(f"[cyan]Bloom Distribution:[/cyan] {cep_config.bloom_distribution}")
    console.print(f"[cyan]Reasoning Traces:[/cyan] {cep_config.enable_reasoning_traces}")
    console.print(f"[cyan]Validation Enabled:[/cyan] {cep_config.enable_validation}")
    if cep_config.enable_validation:
        console.print(f"[cyan]Validator Model:[/cyan] {cep_config.validator_model_id}")
    console.print(f"[cyan]Export JSONL:[/cyan] {export_jsonl}")
    if qa_config.provider == "ollama":
        console.print(f"[cyan]Ollama URL:[/cyan] {qa_config.ollama_url}")
    console.print()

    try:
        run_batch_cep_generation(
            input_dir,
            qa_config.output_dir,
            qa_config,
            cep_config,
            qa_config.workers,
        )

        # Export to JSONL if requested
        if export_jsonl:
            from gtranscriber.schemas import QARecordCEP

            console.print("\n[cyan]Exporting to JSONL format...[/cyan]")
            for json_file in qa_config.output_dir.glob("*_cep_qa.json"):
                try:
                    record = QARecordCEP.load(json_file)
                    jsonl_file = json_file.with_suffix(".jsonl")
                    record.to_jsonl(jsonl_file)
                    console.print(f"  Exported: {jsonl_file.name}")
                except Exception as e:
                    print_warning(f"Failed to export {json_file.name}: {e}")

        print_success("CEP QA generation completed!")

    except Exception as e:
        print_error(f"CEP QA generation failed: {e}")
        raise typer.Exit(code=1) from e


@app.command()
def info() -> None:
    """Display system information and hardware capabilities."""
    import torch

    from gtranscriber.core.hardware import get_device_and_dtype

    console.print("\n[bold]G-Transcriber System Information[/bold]\n")

    # Version
    console.print(f"[cyan]Version:[/cyan] {__version__}")

    # Hardware detection
    hw_config = get_device_and_dtype(force_cpu=False)
    console.print(f"[cyan]Detected Device:[/cyan] {hw_config.device}")
    console.print(f"[cyan]Device Type:[/cyan] {hw_config.device_type.value}")
    console.print(f"[cyan]Data Type:[/cyan] {hw_config.dtype}")

    # PyTorch info
    console.print(f"\n[cyan]PyTorch Version:[/cyan] {torch.__version__}")
    console.print(f"[cyan]CUDA Available:[/cyan] {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        console.print(f"[cyan]CUDA Version:[/cyan] {torch.version.cuda}")
        console.print(f"[cyan]GPU:[/cyan] {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        console.print(f"[cyan]GPU Memory:[/cyan] {gpu_memory:.2f} GB")

    # MPS info
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    console.print(f"[cyan]MPS Available:[/cyan] {has_mps}")

    console.print()


@app.command()
def list_runs(
    pipeline: Annotated[
        str | None,
        typer.Option(
            "--pipeline",
            "-p",
            help="Filter by pipeline type: transcription, qa, cep, kg, evaluation.",
        ),
    ] = None,
    results_dir: Annotated[
        Path,
        typer.Option(
            "--results-dir",
            "-r",
            help="Base results directory. Can be set via GTRANSCRIBER_RESULTS_BASE_DIR env var.",
        ),
    ] = _results_config.base_dir,
) -> None:
    """List all pipeline runs with status and metadata.

    Displays a table of all recorded pipeline runs including status, timing,
    and success rates. Use --pipeline to filter by a specific pipeline type.

    Examples:
        # List all runs
        gtranscriber list-runs

        # List only transcription runs
        gtranscriber list-runs --pipeline transcription

        # Use custom results directory
        gtranscriber list-runs --results-dir /path/to/results
    """
    from rich.table import Table

    from gtranscriber.core.results_manager import ResultsManager
    from gtranscriber.schemas import PipelineType

    # Parse pipeline type if provided
    pipeline_filter = None
    if pipeline:
        try:
            pipeline_filter = PipelineType(pipeline.lower())
        except ValueError:
            valid_types = ", ".join(p.value for p in PipelineType)
            print_error(f"Invalid pipeline type: {pipeline!r}. Valid types: {valid_types}")
            raise typer.Exit(code=1) from None

    # Get runs
    runs = ResultsManager.list_runs(results_dir, pipeline_filter)

    if not runs:
        print_info("No runs found.")
        return

    # Create table
    table = Table(title="Pipeline Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Pipeline", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Started At", style="dim")
    table.add_column("Duration", style="dim")
    table.add_column("Progress", style="green")
    table.add_column("Success Rate", style="yellow")

    for run in runs:
        # Format status with color
        status = run.get("status", "unknown")
        if status == "completed":
            status_styled = "[green]completed[/green]"
        elif status == "failed":
            status_styled = "[red]failed[/red]"
        elif status == "in_progress":
            status_styled = "[yellow]in_progress[/yellow]"
        else:
            status_styled = status

        # Format duration
        duration = run.get("duration_seconds")
        duration_str = f"{duration:.1f}s" if duration is not None else "-"

        # Format progress
        completed = run.get("completed_items", 0)
        total = run.get("total_items", 0)
        progress_str = f"{completed}/{total}" if total else "-"

        # Format success rate
        rate = run.get("success_rate")
        rate_str = f"{rate:.1f}%" if rate is not None else "-"

        # Format started_at
        started = run.get("started_at", "-")
        if started != "-":
            try:
                dt = datetime.fromisoformat(started)
                started = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                pass  # Keep original string if not valid ISO format

        table.add_row(
            run.get("run_id", "unknown"),
            run.get("pipeline_type", "unknown"),
            status_styled,
            started,
            duration_str,
            progress_str,
            rate_str,
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def validate_transcriptions(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing transcription JSON files to validate",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save validated results (if not provided, updates files in-place)",
        ),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            "-t",
            help="Quality threshold (0.0-1.0) for marking transcriptions as valid",
            min=0.0,
            max=1.0,
        ),
    ] = 0.5,
    expected_language: Annotated[
        str,
        typer.Option(
            "--language",
            "-l",
            help="Expected language code (e.g., 'pt', 'en')",
        ),
    ] = "pt",
    report_only: Annotated[
        bool,
        typer.Option(
            "--report-only",
            help="Only display validation report without updating files",
        ),
    ] = False,
) -> None:
    """Validate existing transcriptions for quality issues.

    Detects Whisper failure modes:
    - Wrong language/script (Japanese when expecting Portuguese)
    - Repeated words/phrases
    - Suspicious segment patterns
    - Empty or sparse content

    Examples:
        gtranscriber validate-transcriptions results_tupi/
        gtranscriber validate-transcriptions results/ --threshold 0.6 --report-only
    """
    from gtranscriber.config import TranscriptionQualityConfig

    print_info("Scanning for transcription files...")

    # Find all JSON files in input directory
    json_files = list(input_dir.glob("*_transcription.json"))

    if not json_files:
        print_error(f"No transcription files found in {input_dir}")
        raise typer.Exit(code=1)

    print_info(f"Found {len(json_files)} transcription files")

    # Create quality config with user-specified settings
    quality_config = TranscriptionQualityConfig(
        quality_threshold=threshold,
        expected_language=expected_language,
    )

    # Validate each file
    results = []
    failed_files = []

    with create_progress() as progress:
        task = progress.add_task("Validating...", total=len(json_files))

        for json_path in json_files:
            try:
                # Load record
                with open(json_path) as f:
                    data = json.load(f)
                record = EnrichedRecord(**data)

                # Validate
                validate_enriched_record(record, quality_config)

                results.append(
                    {
                        "file": json_path.name,
                        "valid": record.is_valid,
                        "score": record.transcription_quality.overall_score,
                        "issues": len(record.transcription_quality.issues_detected),
                    }
                )

                # Save updated record if not report-only
                if not report_only:
                    if output_dir:
                        output_path = output_dir / json_path.name
                        output_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        output_path = json_path

                    save_enriched_record(record, output_path)

                if record.is_valid is False:
                    failed_files.append(json_path.name)

            except Exception as e:
                print_error(f"Failed to process {json_path.name}: {e}")
                failed_files.append(json_path.name)

            progress.update(task, advance=1)

    # Display summary
    console.print()
    from rich.table import Table

    table = Table(title="Validation Summary", show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Valid", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Issues", justify="right")

    for result in results:
        valid_icon = "[green]✓[/green]" if result["valid"] else "[red]✗[/red]"
        score_color = "green" if result["score"] >= threshold else "red"
        table.add_row(
            result["file"],
            valid_icon,
            f"[{score_color}]{result['score']:.2f}[/{score_color}]",
            str(result["issues"]),
        )

    console.print(table)
    console.print()

    # Summary statistics
    valid_count = sum(1 for r in results if r["valid"])
    invalid_count = len(results) - valid_count

    console.print(f"[bold]Total files:[/bold] {len(results)}")
    console.print(f"[green]Valid:[/green] {valid_count}")
    console.print(f"[red]Invalid:[/red] {invalid_count}")

    if report_only:
        print_info("Report-only mode: No files were updated")
    elif output_dir:
        print_success(f"Updated files saved to {output_dir}")
    else:
        print_success("Files updated in-place")

    console.print()


@app.command()
def run_info(
    run_id: Annotated[
        str,
        typer.Argument(
            help='Run ID to display, or "latest" for the most recent run.',
        ),
    ],
    pipeline: Annotated[
        str,
        typer.Option(
            "--pipeline",
            "-p",
            help='Pipeline type (required when using "latest").',
        ),
    ] = "transcription",
    results_dir: Annotated[
        Path,
        typer.Option(
            "--results-dir",
            "-r",
            help="Base results directory. Can be set via GTRANSCRIBER_RESULTS_BASE_DIR env var.",
        ),
    ] = _results_config.base_dir,
) -> None:
    """Display detailed information about a specific run.

    Shows complete metadata including execution environment, hardware info,
    configuration snapshot, and processing statistics.

    Examples:
        # Show latest transcription run
        gtranscriber run-info latest --pipeline transcription

        # Show specific run by ID
        gtranscriber run-info 20260204_143052_slurm_grace_1234

        # Show latest QA generation run
        gtranscriber run-info latest --pipeline qa
    """
    from rich.panel import Panel
    from rich.tree import Tree

    from gtranscriber.core.results_manager import ResultsManager
    from gtranscriber.schemas import PipelineType, RunMetadata

    # Parse pipeline type
    try:
        pipeline_type = PipelineType(pipeline.lower())
    except ValueError:
        valid_types = ", ".join(p.value for p in PipelineType)
        print_error(f"Invalid pipeline type: {pipeline!r}. Valid types: {valid_types}")
        raise typer.Exit(code=1) from None

    metadata: RunMetadata | None = None

    if run_id.lower() == "latest":
        # Get latest run for the pipeline
        metadata = ResultsManager.get_latest_run(results_dir, pipeline_type)
        if not metadata:
            print_error(f"No runs found for pipeline: {pipeline}")
            raise typer.Exit(code=1)
    else:
        # Find the specific run
        pipeline_dir = results_dir / pipeline_type.value / run_id
        metadata_path = pipeline_dir / "run_metadata.json"

        if not metadata_path.exists():
            # Try to find in any pipeline directory
            for p in PipelineType:
                test_path = results_dir / p.value / run_id / "run_metadata.json"
                if test_path.exists():
                    metadata = RunMetadata.load(test_path)
                    break

            if not metadata:
                print_error(f"Run not found: {run_id}")
                raise typer.Exit(code=1)
        else:
            metadata = RunMetadata.load(metadata_path)

    # Display metadata
    console.print()

    # Status with color
    status = metadata.status.value
    if status == "completed":
        status_display = "[green]✓ completed[/green]"
    elif status == "failed":
        status_display = "[red]✗ failed[/red]"
    elif status == "in_progress":
        status_display = "[yellow]⏳ in_progress[/yellow]"
    else:
        status_display = status

    # Build tree display
    tree = Tree(f"[bold cyan]{metadata.run_id}[/bold cyan]")

    # Identity
    identity = tree.add("[bold]Identity[/bold]")
    identity.add(f"Pipeline: [magenta]{metadata.pipeline_type.value}[/magenta]")
    identity.add(f"Status: {status_display}")
    identity.add(f"G-Transcriber: v{metadata.gtranscriber_version}")

    # Timing
    timing = tree.add("[bold]Timing[/bold]")
    timing.add(f"Started: {metadata.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if metadata.ended_at:
        timing.add(f"Ended: {metadata.ended_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if metadata.duration_seconds is not None:
        timing.add(f"Duration: {metadata.duration_seconds:.1f} seconds")

    # Progress
    progress = tree.add("[bold]Progress[/bold]")
    progress.add(f"Total items: {metadata.total_items}")
    progress.add(f"Completed: [green]{metadata.completed_items}[/green]")
    progress.add(f"Failed: [red]{metadata.failed_items}[/red]")
    if metadata.success_rate is not None:
        progress.add(f"Success rate: [yellow]{metadata.success_rate:.1f}%[/yellow]")

    # Execution Environment
    env = tree.add("[bold]Execution Environment[/bold]")
    if metadata.execution.is_slurm:
        env.add("Environment: [cyan]SLURM[/cyan]")
        env.add(f"Job ID: {metadata.execution.slurm_job_id}")
        env.add(f"Partition: {metadata.execution.slurm_partition}")
        if metadata.execution.slurm_node:
            env.add(f"Node: {metadata.execution.slurm_node}")
    else:
        env.add("Environment: [cyan]Local[/cyan]")
    env.add(f"Hostname: {metadata.execution.hostname}")
    env.add(f"Username: {metadata.execution.username}")

    # Hardware
    hw = tree.add("[bold]Hardware[/bold]")
    hw.add(f"Device: {metadata.hardware.device_type}")
    if metadata.hardware.gpu_name:
        hw.add(f"GPU: {metadata.hardware.gpu_name}")
    if metadata.hardware.gpu_memory_gb:
        hw.add(f"GPU Memory: {metadata.hardware.gpu_memory_gb} GB")
    if metadata.hardware.cuda_version:
        hw.add(f"CUDA: {metadata.hardware.cuda_version}")
    hw.add(f"CPU Cores: {metadata.hardware.cpu_count}")
    hw.add(f"PyTorch: {metadata.hardware.torch_version}")
    hw.add(f"Python: {metadata.hardware.python_version}")

    # Paths
    paths = tree.add("[bold]Paths[/bold]")
    paths.add(f"Output: {metadata.output_directory}")
    if metadata.checkpoint_file:
        paths.add(f"Checkpoint: {metadata.checkpoint_file}")
    if metadata.input_source:
        paths.add(f"Input source: {metadata.input_source}")

    # Error message if failed
    if metadata.error_message:
        error = tree.add("[bold red]Error[/bold red]")
        error.add(f"[red]{metadata.error_message}[/red]")

    console.print(Panel(tree, title="Run Details", border_style="blue"))
    console.print()


if __name__ == "__main__":
    app()
