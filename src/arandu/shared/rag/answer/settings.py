"""Pydantic settings for the answerer (env prefix ``ARANDU_ANSWERER_``)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnswererSettings(BaseSettings):
    """LLM + budget settings for the Answerer (spec §5.7).

    Held constant across retrieval arms in a single run — this is what
    isolates retrieval-quality from generation-quality in the
    cross-arm comparison (spec §5.1).

    Attributes:
        provider: LLM provider for the answerer. Defaults to ``"ollama"``
            because the project's primary thesis runs use a local
            qwen3:14b; flip to ``"openai"`` for cloud runs.
        model_id: Model identifier; defaults to ``"qwen3:14b"``.
        api_key_env: Env var holding the API key. Ignored for ollama.
        base_url: Base URL override. When ``None``, the per-provider
            default in :class:`LLMClient` is used.
        temperature: Sampling temperature. Default 0.2 — low for
            deterministic answers across reruns (the benchmark must
            be reproducible).
        max_tokens: Max tokens in the answerer's response. Default 1024.
        language: ``"pt"`` or ``"en"``. Selects the prompt template.
        top_k: Maximum passages to consider per question. Defaults to
            10; the actual count may be smaller after token-budget
            packing.
        max_context_tokens: Total context window for the answerer.
            Default 8192. Used by :func:`pack_passages` to compute the
            passage budget.
        prompt_overhead_tokens: Reserved budget for the rendered
            prompt template (everything except the passage list). The
            spec's default of 350 is an empirical estimate for the
            project's prompt; tune downward if larger questions blow
            the budget.
    """

    provider: str = Field(default="ollama")
    model_id: str = Field(default="qwen3:14b")
    api_key_env: str = Field(default="OPENAI_API_KEY")
    base_url: str | None = Field(default=None)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    language: Literal["pt", "en"] = Field(default="pt")
    top_k: int = Field(default=10, ge=1)
    max_context_tokens: int = Field(default=8192, gt=0)
    prompt_overhead_tokens: int = Field(default=350, ge=0)

    model_config = SettingsConfigDict(env_prefix="ARANDU_ANSWERER_", extra="ignore")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        """Lowercase the provider so env-var case (Ollama, OPENAI) doesn't break dispatch."""
        if isinstance(v, str):
            return v.lower()
        return v
