"""Pydantic settings for ``arandu judge-answers`` (env prefix ``ARANDU_JUDGE_ANSWERS_``)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class JudgeAnswersSettings(BaseSettings):
    """LLM + threshold settings for the answer judge (spec §12).

    The judge LLM is **separate** from the Answerer — using a different
    model (or at least a different temperature) reduces the chance that
    the Answerer's own systematic biases are self-validated.

    Attributes:
        provider: LLM provider for the four LLM-based criteria.
            Defaults to ``"ollama"`` to mirror the Answerer's primary
            thesis path. For cloud runs targeting OpenAI proper use
            ``"openai"``; for OpenAI-compatible endpoints (Gemini's
            ``/v1beta/openai/`` compatibility URL, vLLM, etc.) use
            ``"custom"`` and set ``base_url`` accordingly.
        model_id: Model identifier.
        api_key_env: Env var holding the API key. Ignored for ollama.
        base_url: Base URL override. ``None`` lets ``LLMClient`` pick its
            per-provider default; required when ``provider == "custom"``.
        temperature: Sampling temperature. Default 0.3 — slightly higher
            than the answerer's 0.2 to encourage less-anchored judgment.
        max_tokens: Cap on each criterion's response.
        language: Prompt language. Selects ``prompts/judge/criteria/
            <name>/<lang>/prompt.md``.
        abstention_disagreement_audit: If True (default), emit
            ``abstention_audit.jsonl`` for items where the answerer's
            structured ``abstained`` flag disagrees with the abstention
            judge's verdict (the §6.4 disagreement signal). Audit covers
            ~5% of items in the spec's estimate.
    """

    provider: str = Field(default="ollama")
    model_id: str = Field(default="qwen3:14b")
    api_key_env: str = Field(default="OPENAI_API_KEY")
    base_url: str | None = Field(default=None)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    language: Literal["pt", "en"] = Field(default="pt")
    abstention_disagreement_audit: bool = Field(default=True)

    model_config = SettingsConfigDict(env_prefix="ARANDU_JUDGE_ANSWERS_", extra="ignore")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        """Lowercase the provider so env-var case doesn't break LLMProvider() dispatch."""
        if isinstance(v, str):
            return v.lower()
        return v
