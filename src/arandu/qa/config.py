"""Configuration settings for the QA/CEP generation pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class QAConfig(BaseSettings):
    """Configuration settings for the QA generation pipeline.

    Settings are loaded from environment variables with the ARANDU_QA_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARANDU_QA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Provider settings
    provider: str = Field(
        default="ollama",
        description="LLM provider: openai, ollama, custom",
    )
    model_id: str = Field(
        default="qwen3:14b",
        description="Model ID for QA generation",
    )
    ollama_url: str = Field(
        default="http://localhost:11434/v1",
        description="Ollama API base URL",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom base URL for OpenAI-compatible endpoints",
    )

    # Generation settings
    questions_per_document: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of QA pairs to generate per document",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for QA generation LLM",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Max tokens for QA generation LLM",
    )

    # Output settings
    output_dir: Path = Field(
        default=Path("qa_dataset"),
        description="Output directory for QA datasets",
    )

    # Language and prompts
    language: str = Field(
        default="pt",
        description="Language code for QA generation prompts (ISO 639-1: 'en' or 'pt')",
    )

    # Workers (shared setting, loaded from ARANDU_WORKERS)
    workers: int = Field(
        default=2,
        description="Number of parallel workers for QA generation",
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code for QA generation."""
        valid_languages = {"en", "pt"}
        if v not in valid_languages:
            raise ValueError(
                f"Invalid QA language: {v!r}. Must be one of {sorted(valid_languages)}"
            )
        return v


class CEPConfig(BaseSettings):
    """Configuration settings for the CEP (Cognitive Elicitation Pipeline).

    Cognitive scaffolding QA generation based on Bloom's Taxonomy with
    LLM-as-a-Judge validation.

    Settings are loaded from environment variables with the ARANDU_CEP_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARANDU_CEP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Module toggles
    enable_reasoning_traces: bool = Field(
        default=True,
        description="Enable reasoning trace generation for answers",
    )

    # Module I - Bloom Scaffolding settings
    bloom_levels: list[str] = Field(
        default=["remember", "understand", "analyze", "evaluate"],
        description="Bloom levels to use for question generation",
    )
    bloom_distribution: dict[str, float] = Field(
        default={
            "remember": 0.2,
            "understand": 0.3,
            "analyze": 0.3,
            "evaluate": 0.2,
        },
        description="Distribution of questions per Bloom level (must sum to 1.0)",
    )
    enable_scaffolding_context: bool = Field(
        default=True,
        description=(
            "Pass previously generated QA pairs as context to higher Bloom levels. "
            "When enabled, levels are processed in Bloom hierarchy order and each "
            "level receives QA pairs from lower levels in the prompt."
        ),
    )
    max_scaffolding_pairs: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of prior QA pairs to include as scaffolding context",
    )

    # Module II - Reasoning settings
    max_hop_count: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum reasoning hops to detect for multi-hop questions",
    )
    reasoning_max_tokens: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description=(
            "Maximum tokens for reasoning enrichment responses. "
            "Increase for thinking models (Qwen3, DeepSeek-R1) whose <think> "
            "blocks consume tokens before the JSON output."
        ),
    )

    # Scoring weights for judge evaluation
    validation_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum overall score to pass validation",
    )
    faithfulness_weight: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight for faithfulness score in overall calculation",
    )
    bloom_calibration_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for Bloom calibration score in overall calculation",
    )
    informativeness_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for informativeness score in overall calculation",
    )
    self_containedness_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for self-containedness score in overall calculation",
    )

    # Source metadata context
    enable_source_metadata_context: bool = Field(
        default=True,
        description=(
            "Include extracted source metadata (participant name, location, date) "
            "in CEP prompt context to improve contextual grounding."
        ),
    )

    # Language settings
    language: str = Field(
        default="pt",
        description="Language for CEP prompts (ISO 639-1: 'pt' or 'en')",
    )

    @field_validator("bloom_levels")
    @classmethod
    def validate_bloom_levels(cls, v: list[str]) -> list[str]:
        """Validate Bloom taxonomy levels."""
        valid_levels = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
        for level in v:
            if level not in valid_levels:
                raise ValueError(
                    f"Invalid Bloom level: {level!r}. Must be one of {sorted(valid_levels)}"
                )
        return v

    @field_validator("bloom_distribution")
    @classmethod
    def validate_bloom_distribution(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate Bloom distribution sums to 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Bloom distribution must sum to 1.0, got {total}")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code for CEP prompts."""
        valid_languages = {"en", "pt"}
        if v not in valid_languages:
            raise ValueError(
                f"Invalid CEP language: {v!r}. Must be one of {sorted(valid_languages)}"
            )
        return v

    @model_validator(mode="after")
    def validate_scoring_weights(self) -> CEPConfig:
        """Validate that scoring weights sum to 1.0."""
        total = (
            self.faithfulness_weight
            + self.bloom_calibration_weight
            + self.informativeness_weight
            + self.self_containedness_weight
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.3f} "
                f"(faithfulness={self.faithfulness_weight}, "
                f"bloom_calibration={self.bloom_calibration_weight}, "
                f"informativeness={self.informativeness_weight}, "
                f"self_containedness={self.self_containedness_weight})"
            )
        return self


class JudgeConfig(BaseSettings):
    """Configuration settings for the composable judge pipeline.

    Settings are loaded from environment variables with the ARANDU_JUDGE_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARANDU_JUDGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Language for criterion prompts
    language: str = Field(
        default="pt",
        description="Language for judge criterion prompts (ISO 639-1: 'pt' or 'en')",
    )

    # LLM settings for judge (can differ from generator)
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for judge LLM (low for consistent evaluation)",
    )
    max_tokens: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description=(
            "Maximum tokens for judge responses. "
            "Increase for thinking models whose <think> blocks consume tokens."
        ),
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code for judge prompts."""
        valid_languages = {"en", "pt"}
        if v not in valid_languages:
            raise ValueError(
                f"Invalid judge language: {v!r}. Must be one of {sorted(valid_languages)}"
            )
        return v


def get_qa_config() -> QAConfig:
    """Get QA pipeline configuration."""
    return QAConfig()


def get_cep_config() -> CEPConfig:
    """Get CEP (Cognitive Elicitation Pipeline) configuration."""
    return CEPConfig()


def get_judge_config() -> JudgeConfig:
    """Get judge pipeline configuration."""
    return JudgeConfig()
