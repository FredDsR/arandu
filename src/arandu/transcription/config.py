"""Configuration settings for the transcription pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_default_temp_dir() -> str:
    """Get the default temporary directory for the current platform."""
    return str(Path(tempfile.gettempdir()) / "arandu")


class TranscriberConfig(BaseSettings):
    """Configuration settings for the transcription pipeline.

    Settings are loaded from environment variables with the ARANDU_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARANDU_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model settings
    model_id: str = Field(
        default="openai/whisper-large-v3",
        description="Hugging Face model ID for transcription",
    )
    language: str | None = Field(
        default=None,
        description="Language code for transcription (e.g., 'pt'). If None, auto-detect.",
    )
    return_timestamps: bool = Field(
        default=True,
        description="Return timestamps for transcription segments",
    )
    chunk_length_s: int = Field(
        default=30,
        description="Chunk length in seconds for processing",
    )
    stride_length_s: int = Field(
        default=5,
        description="Stride length in seconds between chunks",
    )

    # Hardware settings
    force_cpu: bool = Field(
        default=False,
        description="Force CPU execution instead of GPU",
    )
    quantize: bool = Field(
        default=False,
        description="Enable 8-bit quantization to reduce VRAM usage",
    )
    quantize_bits: int = Field(
        default=8,
        description="Number of bits for quantization",
    )

    # Google Drive settings
    credentials: str = Field(
        default="credentials.json",
        description="Path to Google OAuth2 credentials file",
    )
    token: str = Field(
        default="token.json",
        description="Path to Google OAuth2 token file",
    )
    scopes: list[str] = Field(
        default=["https://www.googleapis.com/auth/drive"],
        description="OAuth2 scopes for Google Drive API",
    )

    # Batch processing settings
    workers: int = Field(
        default=1,
        description="Number of parallel workers for batch processing",
    )
    catalog_file: str = Field(
        default="catalog.csv",
        description="Name of the catalog CSV file",
    )

    # Path settings
    input_dir: str = Field(
        default="./input",
        description="Directory containing input files",
    )
    results_dir: str = Field(
        default="./results",
        description="Directory for transcription results",
    )
    credentials_dir: str = Field(
        default="./",
        description="Directory containing credentials and token files",
    )
    hf_cache_dir: str = Field(
        default="./cache/huggingface",
        description="Hugging Face cache directory for model storage",
    )

    # Processing settings
    temp_dir: str = Field(
        default_factory=_get_default_temp_dir,
        description="Temporary directory for file processing",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for failed operations",
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay in seconds between retry attempts",
    )

    @property
    def credentials_file(self) -> str:
        """Alias for credentials (backward compatibility)."""
        return self.credentials

    @property
    def token_file(self) -> str:
        """Alias for token (backward compatibility)."""
        return self.token


def get_transcriber_config() -> TranscriberConfig:
    """Get transcription pipeline configuration."""
    return TranscriberConfig()
