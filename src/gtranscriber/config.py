"""Configuration module for G-Transcriber."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _get_default_temp_dir() -> str:
    """Get the default temporary directory for the current platform."""
    return str(Path(tempfile.gettempdir()) / "gtranscriber")


@dataclass
class TranscriberConfig:
    """Configuration settings for the transcriber."""

    # Model settings
    model_id: str = "openai/whisper-large-v3"
    return_timestamps: bool = True
    chunk_length_s: int = 30
    stride_length_s: int = 5

    # Hardware settings
    force_cpu: bool = False
    quantize: bool = False
    quantize_bits: int = 8

    # Google Drive settings
    credentials_file: str = "credentials.json"
    token_file: str = "token.json"
    scopes: list[str] = field(default_factory=lambda: ["https://www.googleapis.com/auth/drive"])

    # Processing settings
    temp_dir: str = field(default_factory=_get_default_temp_dir)
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_env(cls) -> TranscriberConfig:
        """Create configuration from environment variables."""
        return cls(
            model_id=os.getenv("GTRANSCRIBER_MODEL_ID", "openai/whisper-large-v3"),
            force_cpu=os.getenv("GTRANSCRIBER_FORCE_CPU", "").lower() == "true",
            quantize=os.getenv("GTRANSCRIBER_QUANTIZE", "").lower() == "true",
            credentials_file=os.getenv("GTRANSCRIBER_CREDENTIALS", "credentials.json"),
            token_file=os.getenv("GTRANSCRIBER_TOKEN", "token.json"),
        )
