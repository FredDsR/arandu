"""Shared helper functions for CLI commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arandu.shared.schemas import TranscriptionSegment

if TYPE_CHECKING:
    from arandu.transcription.engine import TranscriptionResult


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
