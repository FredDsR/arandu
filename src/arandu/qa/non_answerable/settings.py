"""Pydantic settings for non-answerable generation (env ``ARANDU_NONANSWERABLE_``)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class NonAnswerableSettings(BaseSettings):
    """LLM + sampling settings for the non-answerable builder (spec §7, §12).

    Attributes:
        provider: LLM provider. Defaults to ``"ollama"`` (local qwen3:14b
            for thesis runs); flip to ``"openai"`` for cloud.
        model_id: Model identifier; defaults to ``"qwen3:14b"``.
        api_key_env: Env var holding the API key. Ignored for ollama.
        base_url: Base URL override; ``None`` uses the per-provider default.
        language: Prompt language (``"pt"`` project default).
        seeds_per_bloom: Target seeds per Bloom level. 100 -> ~400 total.
        rng_seed: Sampler seed for reproducibility.
        retry_max: Collision retries per seed before skipping it.
        base_temperature: Temperature for the first perturbation attempt;
            each retry adds 0.1 to diversify the sample.
    """

    provider: str = Field(default="ollama")
    model_id: str = Field(default="qwen3:14b")
    api_key_env: str = Field(default="OPENAI_API_KEY")
    base_url: str | None = Field(default=None)
    language: Literal["pt", "en"] = Field(default="pt")
    seeds_per_bloom: int = Field(default=100, ge=1)
    rng_seed: int = Field(default=42)
    retry_max: int = Field(default=3, ge=1)
    base_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    model_config = SettingsConfigDict(env_prefix="ARANDU_NONANSWERABLE_", extra="ignore")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        """Lowercase the provider so env-var case doesn't break dispatch."""
        if isinstance(v, str):
            return v.lower()
        return v
