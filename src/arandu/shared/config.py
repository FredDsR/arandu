"""Shared configuration settings used across multiple pipeline domains."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Shared LLM configuration settings.

    Settings are loaded from environment variables without prefix or with OPENAI_ prefix.
    Used for API keys and shared LLM settings across pipelines.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openai_api_key: str | None = Field(
        default=None,
        alias="OPENAI_API_KEY",
        description="OpenAI API key",
    )

    # Shared base URL (for custom providers)
    base_url: str | None = Field(
        default=None,
        alias="ARANDU_LLM_BASE_URL",
        description="Custom base URL for OpenAI-compatible endpoints",
    )


class ResultsConfig(BaseSettings):
    """Configuration for versioned results management.

    Settings are loaded from environment variables with the ARANDU_RESULTS_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARANDU_RESULTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    base_dir: Path = Field(
        default=Path("./results"),
        description="Base directory for versioned results",
    )
    enable_versioning: bool = Field(
        default=True,
        description="Enable versioned result directories",
    )


def get_llm_config() -> LLMConfig:
    """Get shared LLM configuration."""
    return LLMConfig()


def get_results_config() -> ResultsConfig:
    """Get results versioning configuration."""
    return ResultsConfig()
