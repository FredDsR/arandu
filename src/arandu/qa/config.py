"""Configuration settings for the QA/CEP generation pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from arandu.shared.llm_settings import REASONING_MODEL_MAX_TOKENS

# Upper bound on the per-chunk Bloom ladder (sum of bloom_distribution counts).
# Each pair is one LLM call per chunk, so an unbounded sum would let a typo
# (e.g. --bloom-dist "remember:9999") fan out into a runaway, very expensive run.
MAX_BLOOM_PAIRS_PER_CHUNK = 50


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
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for QA generation LLM",
    )
    max_tokens: int = Field(
        default=REASONING_MODEL_MAX_TOKENS,
        ge=1,
        description=(
            "Max tokens for QA generation LLM. "
            "Sized for thinking models (Qwen3, Gemini 2.5) whose reasoning "
            "tokens consume the budget before the JSON output and would "
            "otherwise truncate it mid-string."
        ),
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
        default=False,
        description="Enable Module II reasoning-trace enrichment (extra LLM call per analyze/"
        "evaluate pair). Off by default: traces/multi-hop were unvalidated self-annotations "
        "unused downstream. The generation 'rationale' field is the inspection aid instead.",
    )

    # Module I - Bloom Scaffolding settings
    bloom_distribution: dict[str, int] = Field(
        default={
            "remember": 3,
            "understand": 1,
            "analyze": 1,
            "evaluate": 1,
        },
        description=(
            "Absolute number of QA pairs to generate at each Bloom level, per "
            "chunk. The keys are the Bloom levels to generate (single source of "
            "truth); the per-chunk ladder size is the sum of these counts (the "
            "document total scales with the chunk count). Locked at 3/1/1/1 for "
            "the thesis run (2026-06-16): remember=3 is the factual base/control "
            "+ Bloom-scaffolding ground; understand/analyze/evaluate=1 each are "
            "the equal-sized cognitive group (balanced factual-vs-cognitive "
            "split). Integer counts, not weights: there is no fractional rounding "
            "to skew the realized split."
        ),
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
        default=REASONING_MODEL_MAX_TOKENS,
        ge=128,
        le=REASONING_MODEL_MAX_TOKENS,
        description=(
            "Maximum tokens for reasoning enrichment responses. "
            "Defaults high for thinking models (Qwen3, DeepSeek-R1, Gemini 2.5) "
            "whose <think> blocks consume tokens before the JSON output and "
            "would otherwise truncate it."
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

    @field_validator("bloom_distribution")
    @classmethod
    def validate_bloom_distribution(cls, v: dict[str, int]) -> dict[str, int]:
        """Validate Bloom distribution counts.

        Each value is the absolute number of pairs to generate at that level
        (per chunk). Counts must reference valid Bloom levels, be non-negative,
        and total between one pair and ``MAX_BLOOM_PAIRS_PER_CHUNK`` (inclusive).
        """
        valid_levels = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
        for level, count in v.items():
            if level not in valid_levels:
                raise ValueError(
                    f"Invalid Bloom level in distribution: {level!r}. "
                    f"Must be one of {sorted(valid_levels)}"
                )
            if count < 0:
                raise ValueError(
                    f"Bloom distribution count for {level!r} must be >= 0, got {count}"
                )
        total = sum(v.values())
        if total < 1:
            raise ValueError(f"Bloom distribution must total at least 1 pair, got {total}")
        if total > MAX_BLOOM_PAIRS_PER_CHUNK:
            raise ValueError(
                f"Bloom distribution must total at most {MAX_BLOOM_PAIRS_PER_CHUNK} pairs "
                f"per chunk, got {total}"
            )
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

    @property
    def pairs_per_chunk(self) -> int:
        """Per-chunk Bloom ladder size (sum of the ``bloom_distribution`` counts).

        Each chunk independently generates this many pairs across the configured
        Bloom levels; the document total scales with the chunk count.
        """
        return sum(self.bloom_distribution.values())


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
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Temperature for judge LLM (low for reproducible single-shot scoring)",
    )
    max_tokens: int = Field(
        default=REASONING_MODEL_MAX_TOKENS,
        ge=128,
        le=REASONING_MODEL_MAX_TOKENS,
        description=(
            "Maximum tokens for judge responses. "
            "Defaults high for thinking models whose <think> blocks consume "
            "tokens before the JSON output and would otherwise truncate it."
        ),
    )

    # Validator client settings (used by TranscriptionJudge LLM stage)
    validator_model: str | None = Field(
        default=None,
        description=(
            "Model ID for the LLM filter stage (language_drift + hallucination_loop), "
            "e.g. 'qwen3:14b' or 'gemini-2.5-flash'. Optional: when unset, "
            "judge-transcription runs in heuristic-only mode and skips the LLM stage."
        ),
    )
    validator_provider: str | None = Field(
        default=None,
        description=(
            "LLM provider for the validator: 'openai', 'ollama', or 'custom'. "
            "Inferred from ARANDU_LLM_BASE_URL (custom when set, else ollama) "
            "when not specified."
        ),
    )
    validator_base_url: str | None = Field(
        default=None,
        description=(
            "Base URL for the validator provider. When unset, "
            "ARANDU_LLM_BASE_URL is inherited only if the resolved provider "
            "is 'custom'; explicit 'openai' or 'ollama' providers keep their "
            "own defaults regardless of ARANDU_LLM_BASE_URL."
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
