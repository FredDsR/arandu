"""Configuration settings for the knowledge graph construction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class KGConfig(BaseSettings):
    """Configuration settings for the knowledge graph construction pipeline.

    Settings are loaded from environment variables with the ARANDU_KG_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARANDU_KG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Backend selection
    backend: str = Field(
        default="atlas",
        pattern="^(atlas)$",
        description="KGC backend: atlas (AutoSchemaKG)",
    )
    backend_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific options passed through to the constructor",
    )

    # LLM Provider settings
    provider: str = Field(
        default="ollama",
        description="LLM provider: openai, ollama, custom",
    )
    model_id: str = Field(
        default="llama3.1:8b",
        description="Model ID for KG construction",
    )
    ollama_url: str = Field(
        default="http://localhost:11434/v1",
        description="Ollama API base URL for KG construction",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom base URL for OpenAI-compatible endpoints",
    )

    # LLM settings
    temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Temperature for KG construction LLM (lower = more consistent)",
    )

    # Language and prompts
    language: str = Field(
        default="pt",
        description="Language code for extraction prompts (ISO 639-1)",
    )

    # Output settings
    output_dir: Path = Field(
        default=Path("knowledge_graphs"),
        description="Output directory for knowledge graphs",
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code for extraction prompts."""
        valid_languages = {"en", "pt"}
        if v not in valid_languages:
            raise ValueError(
                f"Invalid KG language: {v!r}. Must be one of {sorted(valid_languages)}"
            )
        return v


def get_kg_config() -> KGConfig:
    """Get KG pipeline configuration."""
    return KGConfig()
