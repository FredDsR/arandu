"""G-Transcriber CLI interface.

Main entry point for the G-Transcriber application using Typer for CLI
and Rich for visual feedback.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from gtranscriber import __version__
from gtranscriber.core.engine import WhisperEngine
from gtranscriber.core.hardware import get_device_and_dtype
from gtranscriber.core.io import (
    create_temp_file,
    get_output_filename,
    save_enriched_record,
)
from gtranscriber.schemas import EnrichedRecord, InputRecord, TranscriptionSegment
from gtranscriber.utils.logger import (
    console,
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

# Initialize Typer app
app = typer.Typer(
    name="gtranscriber",
    help="Automated transcription system for media files using Whisper ASR.",
    add_completion=False,
    rich_markup_mode="rich",
)


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

    return [
        TranscriptionSegment(
            text=seg.get("text", ""),
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
        )
        for seg in result.segments
    ]


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
        raise typer.Exit()


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
            help="Hugging Face model ID for transcription.",
        ),
    ] = "openai/whisper-large-v3",
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
            help="Enable 8-bit quantization to reduce VRAM usage.",
        ),
    ] = False,
    cpu: Annotated[
        bool,
        typer.Option(
            "--cpu",
            help="Force CPU execution (disables CUDA/MPS even if available).",
        ),
    ] = False,
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
            mimeType=_get_mime_type(file_path),
            parents=[],
            webContentLink="",
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
            help="Hugging Face model ID for transcription.",
        ),
    ] = "openai/whisper-large-v3",
    credentials: Annotated[
        Path,
        typer.Option(
            "--credentials",
            "-c",
            help="Path to Google OAuth2 credentials file.",
        ),
    ] = Path("credentials.json"),
    quantize: Annotated[
        bool,
        typer.Option(
            "--quantize",
            "-q",
            help="Enable 8-bit quantization to reduce VRAM usage.",
        ),
    ] = False,
    cpu: Annotated[
        bool,
        typer.Option(
            "--cpu",
            help="Force CPU execution.",
        ),
    ] = False,
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
        drive_client = DriveClient(credentials_file=str(credentials))

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

        # Cleanup
        temp_file.unlink(missing_ok=True)
        local_output.unlink(missing_ok=True)

    except Exception as e:
        print_error(f"Transcription failed: {e}")
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


def _get_mime_type(file_path: Path) -> str:
    """Get MIME type based on file extension.

    Args:
        file_path: Path to the file.

    Returns:
        MIME type string.
    """
    mime_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/m4a",
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }
    return mime_types.get(file_path.suffix.lower(), "application/octet-stream")


if __name__ == "__main__":
    app()
