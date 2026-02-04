"""Configuration module for G-Transcriber.

Provides separate configuration classes for each pipeline:
- TranscriberConfig: Transcription pipeline settings
- QAConfig: QA generation pipeline settings
- KGConfig: Knowledge graph construction pipeline settings
- EvaluationConfig: Evaluation pipeline settings
"""

from __future__ import annotations

import tempfile
from pathlib import Path

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
        default="llama3.1:8b",
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
    strategies: list[str] = Field(
        default=["factual", "conceptual"],
        description="Question generation strategies: factual, conceptual, temporal, entity",
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
    prompt_path: str | None = Field(
        default=None,
        description="Path to custom prompt templates JSON. If None, uses built-in prompts.",
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

    @field_validator("strategies")
    @classmethod
    def validate_strategies(cls, v: list[str]) -> list[str]:
        """Validate QA generation strategies."""
        valid_strategies = {"factual", "conceptual", "temporal", "entity"}
        for strategy in v:
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Invalid QA strategy: {strategy!r}. Must be one of {sorted(valid_strategies)}"
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

    # Module toggles (progressive adoption)
    enable_bloom_scaffolding: bool = Field(
        default=True,
        description="Enable Bloom taxonomy scaffolding for QA generation",
    )
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

    # Module II - Reasoning settings
    max_hop_count: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum reasoning hops to detect for multi-hop questions",
    )

    # Module III - LLM-as-a-Judge validation settings
    validator_provider: str = Field(
        default="ollama",
        description="LLM provider for validation: openai, ollama, custom",
    )
    validator_model_id: str = Field(
        default="llama3.1:8b",
        description="Model ID for LLM-as-a-Judge validation",
    )
    validator_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for validator (low for consistent evaluation)",
    )
    validation_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum overall score to pass validation",
    )
    faithfulness_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for faithfulness score in overall calculation",
    )
    bloom_calibration_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for Bloom calibration score in overall calculation",
    )
    informativeness_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for informativeness score in overall calculation",
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
            self.faithfulness_weight + self.bloom_calibration_weight + self.informativeness_weight
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.3f} "
                f"(faithfulness={self.faithfulness_weight}, "
                f"bloom_calibration={self.bloom_calibration_weight}, "
                f"informativeness={self.informativeness_weight})"
            )
        return self


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

    # Graph settings
    merge_graphs: bool = Field(
        default=True,
        description="Merge individual graphs into corpus-level graph",
    )
    output_format: str = Field(
        default="graphml",
        pattern="^(graphml|json)$",
        description="Graph export format: graphml (default, NetworkX-compatible) or json",
    )
    schema_mode: str = Field(
        default="dynamic",
        pattern="^(dynamic|predefined)$",
        description="Schema mode: dynamic (infer from data) or predefined",
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
    prompt_path: str = Field(
        default="prompts/pt_prompts.json",
        description="Path to language-specific prompt templates",
    )

    # Output settings
    output_dir: Path = Field(
        default=Path("knowledge_graphs"),
        description="Output directory for knowledge graphs",
    )

    # Workers (shared setting, loaded from GTRANSCRIBER_WORKERS)
    workers: int = Field(
        default=2,
        description="Number of parallel workers for KG construction",
    )


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


def get_transcriber_config() -> TranscriberConfig:
    """Get transcription pipeline configuration."""
    return TranscriberConfig()


def get_qa_config() -> QAConfig:
    """Get QA pipeline configuration."""
    return QAConfig()


def get_cep_config() -> CEPConfig:
    """Get CEP (Cognitive Elicitation Pipeline) configuration."""
    return CEPConfig()


def get_kg_config() -> KGConfig:
    """Get KG pipeline configuration."""
    return KGConfig()


def get_evaluation_config() -> EvaluationConfig:
    """Get evaluation pipeline configuration."""
    return EvaluationConfig()


def get_llm_config() -> LLMConfig:
    """Get shared LLM configuration."""
    return LLMConfig()
