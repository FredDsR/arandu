"""Shared configuration settings used across multiple pipeline domains."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator
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
                    f"Invalid evaluation metric: {metric!r}. "
                    f"Must be one of {sorted(valid_metrics)}"
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


class TranscriptionQualityConfig(BaseSettings):
    """Configuration for transcription quality validation.

    Settings are loaded from environment variables with the ARANDU_QUALITY_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARANDU_QUALITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable transcription quality validation")
    quality_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum quality score to mark transcription as valid",
    )
    expected_language: str = Field(
        default="pt", description="Expected language code (e.g., 'pt', 'en')"
    )

    # Weights (must sum to 1.0, enforced by validator)
    script_match_weight: float = Field(
        default=0.35, description="Weight for script/charset match check"
    )
    repetition_weight: float = Field(default=0.30, description="Weight for repetition detection")
    segment_quality_weight: float = Field(
        default=0.20, description="Weight for segment pattern analysis"
    )
    content_density_weight: float = Field(
        default=0.15, description="Weight for content density check"
    )

    # Thresholds
    max_non_latin_ratio: float = Field(
        default=0.1, description="Maximum ratio of non-Latin characters for Latin languages"
    )
    max_word_repetition_ratio: float = Field(
        default=0.15, description="Maximum ratio of most repeated word"
    )
    max_phrase_repetition_count: int = Field(
        default=4, description="Maximum allowed repetitions of same phrase"
    )
    suspicious_uniform_intervals: int = Field(
        default=5, description="Number of consecutive uniform 1-second intervals to flag"
    )
    min_words_per_minute: float = Field(
        default=30.0, description="Minimum words per minute threshold"
    )
    max_words_per_minute: float = Field(
        default=300.0, description="Maximum words per minute threshold"
    )
    max_empty_segment_ratio: float = Field(
        default=0.2, description="Maximum ratio of empty segments before flagging"
    )
    uniform_interval_tolerance: float = Field(
        default=0.1, description="Tolerance (±seconds) for detecting uniform 1-second intervals"
    )

    @model_validator(mode="after")
    def validate_scoring_weights(self) -> TranscriptionQualityConfig:
        """Validate that scoring weights sum to 1.0."""
        total = (
            self.script_match_weight
            + self.repetition_weight
            + self.segment_quality_weight
            + self.content_density_weight
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Quality scoring weights must sum to 1.0, got {total:.3f}")
        return self


def get_evaluation_config() -> EvaluationConfig:
    """Get evaluation pipeline configuration."""
    return EvaluationConfig()


def get_llm_config() -> LLMConfig:
    """Get shared LLM configuration."""
    return LLMConfig()


def get_results_config() -> ResultsConfig:
    """Get results versioning configuration."""
    return ResultsConfig()


def get_transcription_quality_config() -> TranscriptionQualityConfig:
    """Get transcription quality validation configuration."""
    return TranscriptionQualityConfig()
