"""Shared configuration settings used across multiple pipeline domains."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
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


class EvaluationConfig(BaseSettings):
    """Configuration settings for the evaluation pipeline.

    Settings are loaded from environment variables with the ARANDU_EVAL_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARANDU_EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Metrics to compute
    metrics: list[str] = Field(
        default=["qa", "entity", "relation", "semantic"],
        description="Metrics to compute: qa, entity, relation, semantic",
    )

    # Embedding model for semantic metrics
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for semantic embeddings",
    )

    # Output settings
    output_dir: Path = Field(
        default=Path("evaluation"),
        description="Output directory for evaluation reports",
    )

    # Input directories (can override defaults)
    qa_dir: Path = Field(
        default=Path("qa_dataset"),
        description="Directory containing QA dataset",
    )
    kg_dir: Path = Field(
        default=Path("knowledge_graphs"),
        description="Directory containing knowledge graphs",
    )
    results_dir: Path = Field(
        default=Path("results"),
        description="Directory containing transcription results",
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        """Validate evaluation metric types."""
        valid_metrics = {"qa", "entity", "relation", "semantic"}
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Invalid evaluation metric: {metric!r}. Must be one of {sorted(valid_metrics)}"
                )
        return v


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


def get_evaluation_config() -> EvaluationConfig:
    """Get evaluation pipeline configuration."""
    return EvaluationConfig()


def get_llm_config() -> LLMConfig:
    """Get shared LLM configuration."""
    return LLMConfig()


def get_results_config() -> ResultsConfig:
    """Get results versioning configuration."""
    return ResultsConfig()


