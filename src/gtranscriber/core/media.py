"""Media file utilities for extracting metadata like duration.

Uses ffprobe (from ffmpeg) to extract media duration without requiring
heavy dependencies like moviepy or opencv.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Timeout for ffprobe command in seconds
FFPROBE_TIMEOUT_SECONDS = 30


def has_audio_stream(file_path: str | Path) -> bool:
    """Check if a media file has an audio stream using ffprobe.

    Args:
        file_path: Path to the media file.

    Returns:
        True if the file has at least one audio stream, False otherwise.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",  # Select only audio streams
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
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
        streams = data.get("streams", [])

        return len(streams) > 0

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.warning(f"Failed to check audio streams in {file_path}: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error checking audio streams in {file_path}: {e}")
        return False


def get_audio_stream_info(file_path: str | Path) -> dict | None:
    """Get detailed information about the audio stream(s) in a media file.

    Args:
        file_path: Path to the media file.

    Returns:
        Dictionary with audio stream info, or None if no audio or error.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",  # First audio stream
            "-show_entries",
            "stream=codec_name,codec_type,sample_rate,channels,duration",
            "-of",
            "json",
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
        streams = data.get("streams", [])

        if streams:
            return streams[0]
        return None

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.warning(f"Failed to get audio stream info from {file_path}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error getting audio info from {file_path}: {e}")
        return None


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
