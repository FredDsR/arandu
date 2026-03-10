"""Transcription CLI commands."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.table import Table

from arandu.shared.config import TranscriptionQualityConfig
from arandu.shared.hardware import get_device_and_dtype
from arandu.shared.io import (
    create_temp_file,
    get_mime_type,
    get_output_filename,
    save_enriched_record,
)
from arandu.shared.schemas import EnrichedRecord, InputRecord
from arandu.transcription.config import TranscriberConfig
from arandu.transcription.engine import WhisperEngine
from arandu.transcription.validator import (
    TranscriptionValidator,
    get_quality_issues,
    validate_enriched_record,
)
from arandu.utils.console import console
from arandu.utils.logger import (
    print_error,
    print_info,
    print_success,
    print_warning,
)
from arandu.utils.ui import (
    create_progress,
    display_config_table,
    display_result_panel,
)

from ._helpers import _create_segments_from_result, _safe_int_conversion

logger = logging.getLogger(__name__)

# Load configuration defaults
_config = TranscriberConfig()
DEFAULT_CREDENTIALS_PATH = Path(_config.credentials)
DEFAULT_TOKEN_PATH = Path(_config.token)


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
            help="Hugging Face model ID for transcription. Can be set via ARANDU_MODEL_ID env var.",
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
            "Can be set via ARANDU_QUANTIZE env var.",
        ),
    ] = _config.quantize,
    cpu: Annotated[
        bool,
        typer.Option(
            "--cpu",
            help="Force CPU execution (disables CUDA/MPS even if available). "
            "Can be set via ARANDU_FORCE_CPU env var.",
        ),
    ] = _config.force_cpu,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language code for transcription (e.g., 'pt' for Portuguese). "
            "If not specified, the language will be auto-detected. "
            "Can be set via ARANDU_LANGUAGE env var.",
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
            file_id="local",
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
        issues = get_quality_issues(enriched)
        if issues:
            print_warning(f"Quality issues detected: {issues}")

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
            help="Hugging Face model ID for transcription. Can be set via ARANDU_MODEL_ID env var.",
        ),
    ] = _config.model_id,
    credentials: Annotated[
        Path,
        typer.Option(
            "--credentials",
            "-c",
            help="Path to Google OAuth2 credentials file. "
            "Can be set via ARANDU_CREDENTIALS env var.",
        ),
    ] = DEFAULT_CREDENTIALS_PATH,
    token: Annotated[
        Path,
        typer.Option(
            "--token",
            "-t",
            help="Path to Google OAuth2 token file. Can be set via ARANDU_TOKEN env var.",
        ),
    ] = DEFAULT_TOKEN_PATH,
    quantize: Annotated[
        bool,
        typer.Option(
            "--quantize",
            "-q",
            help="Enable 8-bit quantization to reduce VRAM usage. "
            "Can be set via ARANDU_QUANTIZE env var.",
        ),
    ] = _config.quantize,
    cpu: Annotated[
        bool,
        typer.Option(
            "--cpu",
            help="Force CPU execution. Can be set via ARANDU_FORCE_CPU env var.",
        ),
    ] = _config.force_cpu,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language code for transcription (e.g., 'pt' for Portuguese). "
            "If not specified, the language will be auto-detected. "
            "Can be set via ARANDU_LANGUAGE env var.",
        ),
    ] = _config.language,
) -> None:
    """Transcribe a file from Google Drive.

    Downloads the file, transcribes it, and uploads the result to the same
    Drive folder as the original file.
    """
    from arandu.shared.drive import DriveClient

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
            file_id=metadata["id"],
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
                file_id=input_record.file_id,
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
            issues = get_quality_issues(enriched)
            if issues:
                print_warning(f"Quality issues detected: {issues}")

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
            "Can be set via ARANDU_RESULTS_DIR env var.",
        ),
    ] = Path(_config.results_dir),
    model_id: Annotated[
        str,
        typer.Option(
            "--model-id",
            "-m",
            help="Hugging Face model ID for transcription. Can be set via ARANDU_MODEL_ID env var.",
        ),
    ] = _config.model_id,
    credentials: Annotated[
        Path,
        typer.Option(
            "--credentials",
            "-c",
            help="Path to Google OAuth2 credentials file. "
            "Can be set via ARANDU_CREDENTIALS env var.",
        ),
    ] = DEFAULT_CREDENTIALS_PATH,
    token: Annotated[
        Path,
        typer.Option(
            "--token",
            "-t",
            help="Path to Google OAuth2 token file. Can be set via ARANDU_TOKEN env var.",
        ),
    ] = DEFAULT_TOKEN_PATH,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help="Number of parallel workers (each loads its own model instance). "
            "Can be set via ARANDU_WORKERS env var.",
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
            "Can be set via ARANDU_QUANTIZE env var.",
        ),
    ] = _config.quantize,
    cpu: Annotated[
        bool,
        typer.Option(
            "--cpu",
            help="Force CPU execution. Can be set via ARANDU_FORCE_CPU env var.",
        ),
    ] = _config.force_cpu,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Language code for transcription (e.g., 'pt' for Portuguese). "
            "If not specified, the language will be auto-detected. "
            "Can be set via ARANDU_LANGUAGE env var.",
        ),
    ] = _config.language,
    pipeline_id: Annotated[
        str | None,
        typer.Option(
            "--id",
            help="Pipeline ID for grouping related steps. Auto-generated if omitted.",
        ),
    ] = None,
) -> None:
    """Batch transcribe audio/video files from a catalog.

    Processes all audio and video files listed in the catalog CSV with parallel
    processing support and automatic checkpoint/resume capability. Each worker
    loads its own model instance for true parallel processing.

    The catalog CSV must contain columns: file_id, name, mime_type, size_bytes,
    parents, web_content_link, and optionally duration_milliseconds.

    Transcription results are saved as JSON files in the output directory with
    all metadata including media duration. Progress is automatically checkpointed,
    allowing interrupted jobs to resume from the last completed file.
    """
    from arandu.transcription.batch import BatchConfig, run_batch_transcription

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
        run_batch_transcription(config, pipeline_id=pipeline_id)
        print_success("Batch transcription completed!")

    except Exception as e:
        print_error(f"Batch transcription failed: {e}")
        raise typer.Exit(code=1) from e


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
        arandu validate-transcriptions results_tupi/
        arandu validate-transcriptions results/ --threshold 0.6 --report-only
    """

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

    # Create validator once for reuse across all files
    validator = TranscriptionValidator(quality_config)

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

                # Validate (reuse validator instance)
                validate_enriched_record(record, validator=validator)

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

            except (json.JSONDecodeError, ValidationError, OSError) as e:
                print_error(f"Failed to process {json_path.name}: {e}")
                failed_files.append(json_path.name)

            progress.update(task, advance=1)

    # Display summary
    console.print()

    table = Table(title="Validation Summary", show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("Valid", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Issues", justify="right")

    for result in results:
        valid_icon = "[green]\u2713[/green]" if result["valid"] else "[red]\u2717[/red]"
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
