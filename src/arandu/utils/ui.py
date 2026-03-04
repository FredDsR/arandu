"""Rich UI components for Arandu.

Provides progress bars, panels, and other visual components.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from arandu.utils.console import console

if TYPE_CHECKING:
    from collections.abc import Generator

    from arandu.schemas import EnrichedRecord

# Constants
MAX_DISPLAY_FILES = 20  # Maximum number of files to display in file list


@contextmanager
def create_progress(
    description: str = "Processing",
) -> Generator[Progress]:
    """Create a progress bar context manager.

    Args:
        description: Description for the progress bar.

    Yields:
        Progress instance.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        yield progress


def create_download_progress() -> Progress:
    """Create a progress bar for downloads.

    Returns:
        Progress instance configured for downloads.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Downloading[/bold blue]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )


def create_transcription_progress() -> Progress:
    """Create a progress bar for transcription.

    Returns:
        Progress instance configured for transcription.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Transcribing[/bold green]"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )


def _truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text at word boundary.

    Args:
        text: Text to truncate.
        max_length: Maximum length before truncation.

    Returns:
        Truncated text with ellipsis if needed.
    """
    if len(text) <= max_length:
        return text

    # Find the last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")

    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + "..."


def display_result_panel(record: EnrichedRecord) -> None:
    """Display a panel with transcription results.

    Args:
        record: Enriched record with transcription results.
    """
    truncated_text = _truncate_text(record.transcription_text, 500)

    content = f"""[bold]File:[/bold] {record.name}
[bold]Model:[/bold] {record.model_id}
[bold]Device:[/bold] {record.compute_device}
[bold]Duration:[/bold] {record.processing_duration_sec:.2f}s
[bold]Language:[/bold] {record.detected_language}
[bold]Status:[/bold] {record.transcription_status}

[bold]Transcription:[/bold]
{truncated_text}"""

    panel = Panel(
        content,
        title="[bold green]Transcription Complete[/bold green]",
        border_style="green",
    )
    console.print(panel)


def display_config_table(
    model_id: str,
    device: str,
    quantize: bool,
    source: str,
) -> None:
    """Display a table with current configuration.

    Args:
        model_id: Model ID being used.
        device: Compute device.
        quantize: Whether quantization is enabled.
        source: Source file or folder.
    """
    table = Table(title="Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", model_id)
    table.add_row("Device", device)
    table.add_row("Quantization", "Enabled" if quantize else "Disabled")
    table.add_row("Source", source)

    console.print(table)


def display_file_list(files: list[dict]) -> None:
    """Display a table of files to be processed.

    Args:
        files: List of file metadata dictionaries.
    """
    table = Table(title=f"Found {len(files)} media files")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Size", style="yellow")

    for file in files[:MAX_DISPLAY_FILES]:  # Limit display to first MAX_DISPLAY_FILES
        name = file.get("name", "Unknown")
        mime_type = file.get("mimeType", "Unknown")
        size = file.get("size", "Unknown")
        if isinstance(size, (int, float)):
            size = f"{size / 1024 / 1024:.2f} MB"
        table.add_row(name, mime_type, str(size))

    if len(files) > MAX_DISPLAY_FILES:
        table.add_row("...", f"and {len(files) - MAX_DISPLAY_FILES} more", "")

    console.print(table)
