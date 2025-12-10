"""Media file utilities for extracting metadata like duration.

Uses ffprobe (from ffmpeg) to extract media duration without requiring
heavy dependencies like moviepy or opencv.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Timeout for ffprobe command in seconds
FFPROBE_TIMEOUT_SECONDS = 30


def get_media_duration_ms(file_path: str | Path) -> int | None:
    """Extract media duration in milliseconds using ffprobe.

    Args:
        file_path: Path to the media file.

    Returns:
        Duration in milliseconds, or None if extraction fails.
    """
    try:
        # Use ffprobe to get duration
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(file_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )

        data = json.loads(result.stdout)
        duration_sec = float(data.get("format", {}).get("duration", 0))

        if duration_sec > 0:
            return int(duration_sec * 1000)

        return None

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.warning(f"Failed to extract duration from {file_path}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error extracting duration from {file_path}: {e}")
        return None
