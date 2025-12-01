"""File I/O operations for G-Transcriber.

Handles local file operations and temporary file management.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gtranscriber.schemas import EnrichedRecord


def ensure_temp_dir(base_dir: str = "/tmp/gtranscriber") -> Path:
    """Ensure temporary directory exists.

    Args:
        base_dir: Base directory for temporary files.

    Returns:
        Path to the temporary directory.
    """
    temp_dir = Path(base_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def create_temp_file(
    suffix: str = "",
    prefix: str = "gtranscriber_",
    base_dir: str = "/tmp/gtranscriber",
) -> Path:
    """Create a temporary file.

    Args:
        suffix: File suffix (e.g., '.mp4').
        prefix: File prefix.
        base_dir: Base directory for temporary files.

    Returns:
        Path to the temporary file.
    """
    ensure_temp_dir(base_dir)
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=base_dir)
    # Close the file descriptor as we just need the path
    import os

    os.close(fd)
    return Path(path)


def save_enriched_record(
    record: EnrichedRecord,
    output_path: str | Path,
) -> Path:
    """Save an EnrichedRecord to a JSON file.

    Args:
        record: The enriched record to save.
        output_path: Path to save the JSON file.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict with datetime serialization
    data = record.model_dump()

    # Handle datetime serialization
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return output_path


def get_output_filename(original_name: str, suffix: str = "_transcription.json") -> str:
    """Generate output filename from original file name.

    Args:
        original_name: Original file name.
        suffix: Suffix to append before extension.

    Returns:
        Output filename with suffix.
    """
    path = Path(original_name)
    return f"{path.stem}{suffix}"


def cleanup_temp_files(base_dir: str = "/tmp/gtranscriber") -> int:
    """Clean up temporary files.

    Args:
        base_dir: Base directory for temporary files.

    Returns:
        Number of files cleaned up.
    """
    temp_dir = Path(base_dir)
    if not temp_dir.exists():
        return 0

    count = 0
    for file_path in temp_dir.glob("gtranscriber_*"):
        try:
            file_path.unlink()
            count += 1
        except OSError:
            pass

    return count
