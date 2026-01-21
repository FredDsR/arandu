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

# Timeout for ffmpeg audio extraction in seconds (longer for large files)
FFMPEG_EXTRACT_TIMEOUT_SECONDS = 300

# MIME types that require audio extraction before transcription
VIDEO_MIME_TYPES_REQUIRING_EXTRACTION = {
    "video/mp4",
    "video/quicktime",
    "video/mpeg",
    "video/avi",
    "video/x-msvideo",
    "video/x-matroska",
}


class MediaError(Exception):
    """Base exception for media processing errors."""

    pass


class CorruptedMediaError(MediaError):
    """Raised when a media file is corrupted or incomplete."""

    def __init__(self, file_path: str | Path, reason: str) -> None:
        self.file_path = Path(file_path)
        self.reason = reason
        super().__init__(
            f"Corrupted media file '{self.file_path.name}': {reason}. "
            f"The file may be incomplete, damaged during upload, or recorded incorrectly. "
            f"Please verify the original source file is playable and re-upload if necessary."
        )


class AudioExtractionError(MediaError):
    """Raised when audio extraction from a media file fails."""

    def __init__(self, file_path: str | Path, reason: str) -> None:
        self.file_path = Path(file_path)
        self.reason = reason
        super().__init__(
            f"Failed to extract audio from '{self.file_path.name}': {reason}. "
            f"Ensure ffmpeg is installed and the file contains a valid audio stream."
        )


class UnsupportedCodecError(MediaError):
    """Raised when a media file has an unsupported audio codec."""

    def __init__(self, file_path: str | Path, codec: str) -> None:
        self.file_path = Path(file_path)
        self.codec = codec
        super().__init__(
            f"Unsupported audio codec '{codec}' in '{self.file_path.name}'. "
            f"The file's audio format is not supported by the transcription backend."
        )


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
        logger.warning(f"Failed to extract duration from {file_path}: {e}", exc_info=True)
        return None
    except Exception:
        logger.exception(f"Unexpected error extracting duration from {file_path}")
        return None


def validate_media_file(file_path: str | Path) -> None:
    """Validate that a media file is not corrupted.

    Checks for common corruption issues like missing moov atom in MP4 files.

    Args:
        file_path: Path to the media file.

    Raises:
        CorruptedMediaError: If the file is corrupted or incomplete.
    """
    file_path = Path(file_path)

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_format",
            "-of", "json",
            str(file_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )

        # Check for common corruption indicators in stderr
        stderr = result.stderr.lower()

        if "moov atom not found" in stderr:
            raise CorruptedMediaError(
                file_path,
                "Missing moov atom (MP4 metadata header). "
                "The recording may have been interrupted or the file was not properly finalized"
            )

        if "invalid data found" in stderr:
            raise CorruptedMediaError(
                file_path,
                "Invalid data structure. The file format is damaged or unrecognizable"
            )

        if result.returncode != 0:
            # Generic corruption if ffprobe fails
            error_detail = result.stderr.strip() if result.stderr else "Unknown error"
            raise CorruptedMediaError(file_path, error_detail)

    except subprocess.TimeoutExpired:
        raise CorruptedMediaError(
            file_path,
            "File analysis timed out - file may be extremely large or corrupted"
        )
    except CorruptedMediaError:
        raise
    except Exception as e:
        logger.warning(f"Unexpected error validating {file_path}: {e}")
        # Don't raise for unexpected errors, let downstream processing handle it


def requires_audio_extraction(mime_type: str) -> bool:
    """Check if a MIME type requires audio extraction before transcription.

    Args:
        mime_type: The MIME type of the file.

    Returns:
        True if audio should be extracted, False if file can be processed directly.
    """
    return mime_type in VIDEO_MIME_TYPES_REQUIRING_EXTRACTION


def extract_audio(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """Extract audio from a media file using ffmpeg.

    Converts audio to WAV format optimized for speech recognition:
    - 16kHz sample rate (Whisper's native rate)
    - Mono channel
    - 16-bit PCM

    Args:
        input_path: Path to the input media file (video or audio).
        output_path: Path for the output WAV file.
        sample_rate: Output sample rate in Hz (default: 16000 for Whisper).
        mono: Convert to mono audio (default: True).

    Returns:
        Path to the extracted audio file.

    Raises:
        CorruptedMediaError: If the input file is corrupted.
        AudioExtractionError: If audio extraction fails.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # First validate the input file
    validate_media_file(input_path)

    # Check for audio stream
    if not has_audio_stream(input_path):
        raise AudioExtractionError(
            input_path,
            "No audio stream found. The file may be a silent video or have an unsupported audio codec"
        )

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-i", str(input_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", str(sample_rate),  # Sample rate
    ]

    if mono:
        cmd.extend(["-ac", "1"])  # Mono

    cmd.append(str(output_path))

    logger.debug(f"Extracting audio: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_EXTRACT_TIMEOUT_SECONDS,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()

            # Parse common ffmpeg errors for better messages
            if "moov atom not found" in stderr.lower():
                raise CorruptedMediaError(
                    input_path,
                    "Missing moov atom (MP4 metadata header)"
                )
            elif "no such file" in stderr.lower():
                raise AudioExtractionError(input_path, "Input file not found")
            elif "invalid data" in stderr.lower():
                raise CorruptedMediaError(input_path, "Invalid data structure")
            elif "does not contain any stream" in stderr.lower():
                raise AudioExtractionError(input_path, "No audio stream in file")
            else:
                # Generic extraction error
                error_msg = stderr[:200] if len(stderr) > 200 else stderr
                raise AudioExtractionError(input_path, error_msg or "Unknown ffmpeg error")

        # Verify output file was created and has content
        if not output_path.exists():
            raise AudioExtractionError(input_path, "Output file was not created")

        if output_path.stat().st_size == 0:
            output_path.unlink()
            raise AudioExtractionError(input_path, "Extracted audio file is empty")

        logger.info(
            f"Extracted audio: {input_path.name} -> {output_path.name} "
            f"({output_path.stat().st_size / 1024:.1f} KB)"
        )

        return output_path

    except subprocess.TimeoutExpired:
        # Clean up partial output
        if output_path.exists():
            output_path.unlink()
        raise AudioExtractionError(
            input_path,
            f"Audio extraction timed out after {FFMPEG_EXTRACT_TIMEOUT_SECONDS} seconds"
        )
    except (CorruptedMediaError, AudioExtractionError):
        raise
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        raise AudioExtractionError(input_path, str(e))
