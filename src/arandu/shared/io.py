"""File I/O operations for Arandu.

Handles local file operations and temporary file management.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arandu.shared.schemas import EnrichedRecord


def _get_default_temp_dir() -> str:
    """Get the default temporary directory for the current platform.

    Returns:
        Cross-platform temporary directory path.
    """
    return str(Path(tempfile.gettempdir()) / "arandu")


def ensure_temp_dir(base_dir: str | None = None) -> Path:
    """Ensure temporary directory exists.

    Args:
        base_dir: Base directory for temporary files. Uses system temp dir if None.

    Returns:
        Path to the temporary directory.
    """
    if base_dir is None:
        base_dir = _get_default_temp_dir()
    temp_dir = Path(base_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def create_temp_file(
    suffix: str = "",
    prefix: str = "arandu_",
    base_dir: str | None = None,
) -> Path:
    """Create a temporary file.

    Args:
        suffix: File suffix (e.g., '.mp4').
        prefix: File prefix.
        base_dir: Base directory for temporary files. Uses system temp dir if None.

    Returns:
        Path to the temporary file.

    Note:
        Uses NamedTemporaryFile with delete=False for better safety and to avoid race conditions.
    """
    if base_dir is None:
        base_dir = _get_default_temp_dir()
    ensure_temp_dir(base_dir)
    # Use NamedTemporaryFile with delete=False for better safety and to avoid race conditions
    with tempfile.NamedTemporaryFile(
        suffix=suffix, prefix=prefix, dir=base_dir, delete=False
    ) as tmp:
        path = tmp.name
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

    # Use Pydantic's built-in JSON serialization for nested models and datetimes
    json_str = record.model_dump_json(indent=2)
    output_path.write_text(json_str, encoding="utf-8")
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


def get_mime_type(file_path: Path) -> str:
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


def cleanup_temp_files(base_dir: str | None = None) -> tuple[int, int]:
    """Clean up temporary files.

    Args:
        base_dir: Base directory for temporary files. Uses system temp dir if None.

    Returns:
        Tuple of (success_count, failure_count).
    """
    if base_dir is None:
        base_dir = _get_default_temp_dir()
    temp_dir = Path(base_dir)
    if not temp_dir.exists():
        return (0, 0)

    success_count = 0
    failure_count = 0
    for file_path in temp_dir.glob("arandu_*"):
        try:
            file_path.unlink()
            success_count += 1
        except OSError as e:
            import logging

            logging.warning(f"Failed to delete temporary file {file_path}: {e}")
            failure_count += 1

    return (success_count, failure_count)
