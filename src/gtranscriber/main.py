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
from gtranscriber.config import TranscriberConfig
from gtranscriber.core.engine import WhisperEngine
from gtranscriber.core.hardware import get_device_and_dtype
from gtranscriber.core.io import (
    create_temp_file,
    get_mime_type,
    get_output_filename,
    save_enriched_record,
)
from gtranscriber.schemas import EnrichedRecord, InputRecord, TranscriptionSegment
from gtranscriber.utils.console import console
from gtranscriber.utils.logger import (
    print_error,
    print_info,
    print_success,
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
        return int(value)
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

    # Display configuration
    console.print("\n[bold]QA Generation Configuration[/bold]\n")
    console.print(f"[cyan]Input Directory:[/cyan] {input_dir}")
    console.print(f"[cyan]Output Directory:[/cyan] {qa_config.output_dir}")
    console.print(f"[cyan]Provider:[/cyan] {qa_config.provider}")
    console.print(f"[cyan]Model:[/cyan] {qa_config.model_id}")
    console.print(f"[cyan]Workers:[/cyan] {qa_config.workers}")
    console.print(f"[cyan]Questions per document:[/cyan] {qa_config.questions_per_document}")
    console.print(f"[cyan]Strategies:[/cyan] {', '.join(qa_config.strategies)}")
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


if __name__ == "__main__":
    app()
