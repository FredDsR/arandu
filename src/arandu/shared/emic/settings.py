"""Pydantic settings for ``arandu emic-prepass`` (env prefix ``ARANDU_EMIC_PREPASS_``)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmicPrepassSettings(BaseSettings):
    """LLM settings for the ordinal emic-validity pre-pass (spec §5).

    The pre-pass runs the ``emic_validity`` ordinal criterion over the
    canonical-approved CEP pairs to produce per-pair scores that feed the
    stratified sample. The score is a sampling aid, **not** ground truth (the
    human annotators are the reference), so a modest model is fine.

    Attributes:
        provider: LLM provider (``ollama`` default; ``openai`` or ``custom``
            for cloud / OpenAI-compatible endpoints).
        model_id: Model identifier.
        api_key_env: Env var holding the API key (ignored for ollama).
        base_url: Base URL override; required when ``provider == "custom"``.
        temperature: Sampling temperature. Default 0.1 — the emic judgment is
            structural, not creative (spec §4.2 principle 8).
        max_tokens: Cap on each criterion response.
        language: Prompt language; selects
            ``prompts/judge/criteria/emic_validity/<lang>/prompt.md``.
            Only ``pt`` ships today.
    """

    provider: str = Field(default="ollama")
    model_id: str = Field(default="qwen3:14b")
    api_key_env: str = Field(default="OPENAI_API_KEY")
    base_url: str | None = Field(default=None)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    language: Literal["pt"] = Field(default="pt")

    model_config = SettingsConfigDict(env_prefix="ARANDU_EMIC_PREPASS_", extra="ignore")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        """Lowercase the provider so env-var case doesn't break dispatch."""
        return v.lower() if isinstance(v, str) else v
