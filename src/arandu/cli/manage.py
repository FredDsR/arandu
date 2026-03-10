"""Management CLI commands: list-runs, run-info, replicate, etc."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from arandu import __version__
from arandu.shared.config import ResultsConfig
from arandu.shared.io import save_enriched_record
from arandu.shared.schemas import EnrichedRecord
from arandu.utils.console import console
from arandu.utils.logger import (
    print_error,
    print_info,
    print_success,
    print_warning,
    setup_logging,
)

_results_config = ResultsConfig()


def refresh_auth(
    credentials: Annotated[
        Path,
        typer.Option("--credentials", "-c", help="Path to Google OAuth2 credentials file."),
    ] = Path("credentials.json"),
    token: Annotated[
        Path,
        typer.Option("--token", "-t", help="Path to token file to refresh."),
    ] = Path("token.json"),
) -> None:
    """Fully refresh Google OAuth2 authentication token.

    This command deletes the existing token file and initiates a fresh OAuth2
    authorization flow. Use this when you need to:

    - Re-authorize with different Google account
    - Fix authentication issues or permission problems
    - Update token after revoking access in Google Account settings
    """
    from arandu.shared.drive import DriveClient

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


def replicate(
    source_id: Annotated[
        str,
        typer.Argument(help="Pipeline ID to replicate from."),
    ],
    target_id: Annotated[
        str | None,
        typer.Option("--id", help="New pipeline ID. Auto-generated if omitted."),
    ] = None,
    results_dir: Annotated[
        Path,
        typer.Option(
            "--results-dir",
            "-r",
            help="Base results directory. Can be set via ARANDU_RESULTS_BASE_DIR env var.",
        ),
    ] = _results_config.base_dir,
) -> None:
    """Replicate (clone) an existing pipeline run to a new ID.

    Copies the entire pipeline directory tree and rewrites metadata with a new
    pipeline ID. This enables experimentation workflows: clone a completed run,
    then re-run only specific steps (e.g., QA with new prompts) on top of the
    cloned outputs.

    Examples:
        # Auto-generate new ID
        arandu replicate 20260204_143052_local

        # Specify target ID
        arandu replicate 20260204_143052_local --id my-experiment-v2
    """
    from rich.panel import Panel

    from arandu.shared.results_manager import ResultsManager
    from arandu.shared.schemas import PipelineMetadata

    try:
        new_id = ResultsManager.replicate_pipeline(
            base_dir=results_dir,
            source_pipeline_id=source_id,
            target_pipeline_id=target_id,
        )

        # Load replicated pipeline metadata for summary
        pipeline_meta = PipelineMetadata.load(results_dir / new_id / "pipeline.json")

        summary_lines = [
            f"[cyan]New Pipeline ID:[/cyan] {new_id}",
            f"[cyan]Source Pipeline:[/cyan] {source_id}",
            f"[cyan]Steps Copied:[/cyan] {', '.join(pipeline_meta.steps_run)}",
            f"[cyan]Location:[/cyan] {results_dir / new_id}",
        ]

        console.print()
        console.print(Panel("\n".join(summary_lines), title="Replication Complete"))
        console.print()

        print_success(f"Pipeline replicated: {source_id} -> {new_id}")

    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:
        print_error(f"Replication failed: {e}")
        raise typer.Exit(code=1) from e


def info() -> None:
    """Display system information and hardware capabilities."""
    import torch

    from arandu.shared.hardware import get_device_and_dtype

    console.print("\n[bold]Arandu System Information[/bold]\n")

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
            help="Base results directory. Can be set via ARANDU_RESULTS_BASE_DIR env var.",
        ),
    ] = _results_config.base_dir,
) -> None:
    """List all pipeline runs with status and metadata.

    Displays a table of all recorded pipeline runs including status, timing,
    and success rates. Use --pipeline to filter by a specific pipeline type.

    Examples:
        # List all runs
        arandu list-runs

        # List only transcription runs
        arandu list-runs --pipeline transcription

        # Use custom results directory
        arandu list-runs --results-dir /path/to/results
    """
    from rich.table import Table

    from arandu.shared.results_manager import ResultsManager
    from arandu.shared.schemas import PipelineType

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
    table.add_column("Pipeline ID", style="cyan")
    table.add_column("Step", style="magenta")
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
            run.get("pipeline_id") or run.get("run_id", "unknown"),
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


def run_info(
    run_id: Annotated[
        str,
        typer.Argument(help='Run ID to display, or "latest" for the most recent run.'),
    ],
    pipeline: Annotated[
        str,
        typer.Option("--pipeline", "-p", help='Pipeline type (required when using "latest").'),
    ] = "transcription",
    results_dir: Annotated[
        Path,
        typer.Option(
            "--results-dir",
            "-r",
            help="Base results directory. Can be set via ARANDU_RESULTS_BASE_DIR env var.",
        ),
    ] = _results_config.base_dir,
) -> None:
    """Display detailed information about a specific run.

    Shows complete metadata including execution environment, hardware info,
    configuration snapshot, and processing statistics.

    Examples:
        # Show latest transcription run
        arandu run-info latest --pipeline transcription

        # Show specific run by ID
        arandu run-info 20260204_143052_slurm_grace_1234

        # Show latest QA generation run
        arandu run-info latest --pipeline qa
    """
    from rich.panel import Panel
    from rich.tree import Tree

    from arandu.shared.results_manager import ResultsManager
    from arandu.shared.schemas import PipelineType, RunMetadata

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
        # ID-first layout: results/{run_id}/{pipeline_type}/run_metadata.json
        metadata_path = results_dir / run_id / pipeline_type.value / "run_metadata.json"

        if not metadata_path.exists():
            # Try to find in any step directory under this pipeline ID
            for p in PipelineType:
                test_path = results_dir / run_id / p.value / "run_metadata.json"
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
        status_display = "[green]\u2713 completed[/green]"
    elif status == "failed":
        status_display = "[red]\u2717 failed[/red]"
    elif status == "in_progress":
        status_display = "[yellow]\u23f3 in_progress[/yellow]"
    else:
        status_display = status

    # Build tree display
    tree = Tree(f"[bold cyan]{metadata.run_id}[/bold cyan]")

    # Identity
    identity = tree.add("[bold]Identity[/bold]")
    identity.add(f"Pipeline: [magenta]{metadata.pipeline_type.value}[/magenta]")
    identity.add(f"Status: {status_display}")
    identity.add(f"Arandu: v{metadata.arandu_version}")

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


def rebuild_index(
    results_dir: Annotated[
        Path,
        typer.Option(
            "--results-dir",
            "-r",
            help="Base results directory. Can be set via ARANDU_RESULTS_BASE_DIR env var.",
        ),
    ] = _results_config.base_dir,
) -> None:
    """Rebuild index.json from existing run directories.

    Scans all pipeline ID directories for run_metadata.json files and rebuilds
    the global index.json.

    Examples:
        # Rebuild index in default results directory
        arandu rebuild-index

        # Rebuild index in custom results directory
        arandu rebuild-index --results-dir /path/to/results
    """
    from arandu.shared.results_manager import ResultsManager
    from arandu.shared.schemas import PipelineType, RunMetadata

    base_dir = results_dir.resolve()
    if not base_dir.exists():
        print_error(f"Results directory not found: {base_dir}")
        raise typer.Exit(code=1)

    # Collect all runs by scanning ID-first layout: results/{pipeline_id}/{step}/
    all_metadata: list[tuple[PipelineType, RunMetadata, Path]] = []

    for pipeline_dir in sorted(base_dir.iterdir()):
        if not pipeline_dir.is_dir():
            continue
        # Skip non-pipeline directories
        if not (pipeline_dir / "pipeline.json").exists():
            continue

        for p_type in PipelineType:
            step_dir = pipeline_dir / p_type.value
            metadata_path = step_dir / "run_metadata.json"
            if not metadata_path.exists():
                continue
            try:
                metadata = RunMetadata.load(metadata_path)
                all_metadata.append((p_type, metadata, step_dir))
            except Exception as e:
                print_warning(f"Skipping {pipeline_dir.name}/{p_type.value}: {e}")

    if not all_metadata:
        print_warning("No runs found to rebuild from.")
        return

    # Delete stale index so we rebuild from scratch
    index_path = base_dir / "index.json"
    if index_path.exists():
        index_path.unlink()
        print_info("Removed stale index.json")

    # Rebuild index
    total_runs = 0
    for p_type, metadata, step_dir in sorted(all_metadata, key=lambda t: t[1].started_at):
        manager = ResultsManager(base_dir, p_type)
        manager._run_dir = step_dir
        manager._metadata = metadata
        manager._update_index()
        total_runs += 1

    # Summarise per pipeline type
    from collections import Counter

    step_counts = Counter(p.value for p, _, _ in all_metadata)
    for step_name, count in step_counts.items():
        print_info(f"{step_name}: {count} run(s)")

    print_success(f"Rebuilt index.json ({total_runs} runs)")


def enrich_metadata(
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
    pipeline_id: Annotated[
        str | None,
        typer.Option("--pipeline-id", "--id", help="Pipeline ID for versioned results resolution."),
    ] = None,
) -> None:
    """Enrich existing transcription JSONs with source metadata.

    Reads the catalog CSV, matches rows to transcription files by file_id,
    and extracts structured metadata (participant, location, date, etc.)
    from filenames and folder paths.

    Examples:
        arandu enrich-metadata results/outputs/ input/catalog.csv

        arandu enrich-metadata results/ input/catalog.csv --id 20250101_120000
    """
    import csv

    from arandu.metadata import GDriveCatalogExtractor, enrich_with_source_metadata
    from arandu.qa.batch import _resolve_transcription_dir

    setup_logging()

    # Resolve transcription directory
    try:
        transcription_dir = _resolve_transcription_dir(input_dir, pipeline_id)
    except Exception:
        transcription_dir = input_dir

    transcription_files = list(transcription_dir.glob("*_transcription.json"))
    if not transcription_files:
        print_error(f"No transcription files found in {transcription_dir}")
        raise typer.Exit(code=1)

    print_info(f"Found {len(transcription_files)} transcription file(s)")

    # Build file_id -> row lookup from catalog
    catalog_lookup: dict[str, dict[str, str]] = {}
    with open(catalog, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row.get("file_id") or row.get("gdrive_id", "")
            if file_id:
                catalog_lookup[file_id] = dict(row)

    print_info(f"Loaded {len(catalog_lookup)} catalog entries")

    extractor = GDriveCatalogExtractor()
    enriched_count = 0
    skipped_count = 0

    for tf in transcription_files:
        try:
            record = EnrichedRecord.model_validate_json(tf.read_text(encoding="utf-8"))
        except (ValidationError, Exception) as e:
            print_warning(f"Skipping {tf.name}: {e}")
            skipped_count += 1
            continue

        catalog_row = catalog_lookup.get(record.file_id)
        if catalog_row is None:
            print_warning(f"No catalog entry for {record.file_id} ({record.name})")
            skipped_count += 1
            continue

        enrich_with_source_metadata(record, catalog_row, extractor=extractor)
        save_enriched_record(record, tf)
        enriched_count += 1

    print_success(f"Enriched {enriched_count} transcription(s) (skipped {skipped_count})")
