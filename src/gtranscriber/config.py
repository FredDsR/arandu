"""Configuration module for G-Transcriber.

Provides separate configuration classes for each pipeline:
- TranscriberConfig: Transcription pipeline settings
- QAConfig: QA/CEP generation pipeline settings
- CEPConfig: Cognitive Elicitation Pipeline (Bloom scaffolding + validation)
- KGConfig: Knowledge graph construction pipeline settings
- EvaluationConfig: Evaluation pipeline settings
- LLMConfig: Shared LLM/API key settings
- ResultsConfig: Versioned results management settings
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_default_temp_dir() -> str:
    """Get the default temporary directory for the current platform."""
    return str(Path(tempfile.gettempdir()) / "gtranscriber")


class TranscriberConfig(BaseSettings):
    """Configuration settings for the transcription pipeline.

    Settings are loaded from environment variables with the GTRANSCRIBER_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GTRANSCRIBER_",
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


class QAConfig(BaseSettings):
    """Configuration settings for the QA generation pipeline.

    Settings are loaded from environment variables with the GTRANSCRIBER_QA_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GTRANSCRIBER_QA_",
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

    # Workers (shared setting, loaded from GTRANSCRIBER_WORKERS)
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

    Settings are loaded from environment variables with the GTRANSCRIBER_CEP_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GTRANSCRIBER_CEP_",
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
    enable_validation: bool = Field(
        default=True,
        description="Enable LLM-as-a-Judge validation (requires additional LLM calls)",
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

    # Module III - LLM-as-a-Judge validation settings
    validator_provider: str = Field(
        default="ollama",
        description="LLM provider for validation: openai, ollama, custom",
    )
    validator_model_id: str = Field(
        default="qwen3:14b",
        description="Model ID for LLM-as-a-Judge validation",
    )
    validator_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for validator (low for consistent evaluation)",
    )
    validator_max_tokens: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description=(
            "Maximum tokens for validation responses. "
            "Increase for thinking models (Qwen3, DeepSeek-R1) whose <think> "
            "blocks consume tokens before the JSON output."
        ),
    )
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

    Settings are loaded from environment variables with the GTRANSCRIBER_JUDGE_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GTRANSCRIBER_JUDGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Enable composable judge pipeline
    use_composable_pipeline: bool = Field(
        default=True,
        description=(
            "Use composable G-Eval-style judge pipeline (one criterion per LLM call). "
            "When False, falls back to legacy single-call validation."
        ),
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


class KGConfig(BaseSettings):
    """Configuration settings for the knowledge graph construction pipeline.

    Settings are loaded from environment variables with the GTRANSCRIBER_KG_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GTRANSCRIBER_KG_",
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

    # Workers
    workers: int = Field(
        default=2,
        description="Number of parallel workers for KG construction",
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


class EvaluationConfig(BaseSettings):
    """Configuration settings for the evaluation pipeline.

    Settings are loaded from environment variables with the GTRANSCRIBER_EVAL_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GTRANSCRIBER_EVAL_",
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
        alias="GTRANSCRIBER_LLM_BASE_URL",
        description="Custom base URL for OpenAI-compatible endpoints",
    )


class ResultsConfig(BaseSettings):
    """Configuration for versioned results management.

    Settings are loaded from environment variables with the GTRANSCRIBER_RESULTS_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GTRANSCRIBER_RESULTS_",
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

    Settings are loaded from environment variables with the GTRANSCRIBER_QUALITY_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GTRANSCRIBER_QUALITY_",
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


def get_transcriber_config() -> TranscriberConfig:
    """Get transcription pipeline configuration."""
    return TranscriberConfig()


def get_qa_config() -> QAConfig:
    """Get QA pipeline configuration."""
    return QAConfig()


def get_cep_config() -> CEPConfig:
    """Get CEP (Cognitive Elicitation Pipeline) configuration."""
    return CEPConfig()


def get_judge_config() -> JudgeConfig:
    """Get judge pipeline configuration."""
    return JudgeConfig()


def get_kg_config() -> KGConfig:
    """Get KG pipeline configuration."""
    return KGConfig()


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
