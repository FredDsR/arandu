"""Canonical LLM settings shared across pipeline stages.

Defines :class:`LLMSettings`, the single source of truth for the connection +
sampling fields every LLM-driven stage needs. Kept separate from
``llm_client`` (the SDK wrapper) so the config layer doesn't live inside the
client module; the settings -> client bridge is
:func:`arandu.shared.llm_client.build_llm_client_from_settings`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Token headroom sized for reasoning models (qwen3:14b, gemini-2.5-flash)
#: whose internal thinking tokens count against the response budget. A tighter
#: cap is exhausted mid-reasoning and truncates the (often JSON) response,
#: surfacing downstream as a parse failure and a dropped result. Single source
#: for the value every LLM stage uses, so they all move together; referenced as
#: the :class:`LLMSettings` ``max_tokens`` default and re-exported as
#: ``arandu.shared.judge.criterion.DEFAULT_MAX_TOKENS``.
REASONING_MODEL_MAX_TOKENS = 8192


class LLMSettings(BaseSettings):
    """Canonical LLM connection + sampling settings for any pipeline stage.

    Single source of truth for the fields every LLM-driven stage needs to
    build an :class:`~arandu.shared.llm_client.LLMClient`. Stages with extra
    domain config (token budgets, criterion thresholds, sampler seeds) subclass
    this and add their own fields, overriding only the LLM defaults they
    deliberately change and pinning a per-stage ``env_prefix`` (so each stage
    can be configured independently, e.g. a cheaper model for one stage than
    another). Subclasses inherit ``extra="ignore"`` and the provider
    normalizer; they only need to set ``env_prefix``.

    Attributes:
        provider: LLM provider (``"ollama"`` default; ``"openai"`` for OpenAI
            proper; ``"custom"`` for an OpenAI-compatible endpoint, paired
            with ``base_url``).
        model_id: Model identifier.
        api_key_env: Env var holding the API key (ignored for ollama).
        base_url: Base URL override; required when ``provider == "custom"``.
        temperature: Sampling temperature.
        max_tokens: Cap on each response.
        language: Prompt language; selects the per-stage prompt template.
        workers: Client-side concurrent LLM requests for batch runners
            wired through :func:`arandu.utils.concurrency.map_concurrent`
            (answer, judge-answers); other stages ignore it. Pair with
            matching server slots (``OLLAMA_NUM_PARALLEL``) and the
            per-slot context VRAM budget (``scripts/slurm/rag/*.slurm``).
    """

    provider: str = Field(default="ollama")
    model_id: str = Field(default="qwen3:14b")
    api_key_env: str = Field(default="OPENAI_API_KEY")
    base_url: str | None = Field(default=None)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # See REASONING_MODEL_MAX_TOKENS. Every stage inherits this headroom; even
    # short-answer stages (e.g. the answerer) need it because the response is
    # structured and reasoning models burn thinking tokens against the budget.
    max_tokens: int = Field(default=REASONING_MODEL_MAX_TOKENS, gt=0)
    language: Literal["pt", "en"] = Field(default="pt")
    workers: int = Field(default=1, ge=1, le=16)

    model_config = SettingsConfigDict(env_prefix="ARANDU_LLM_", extra="ignore")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        """Lowercase the provider so env-var case (Ollama, OPENAI) doesn't break dispatch."""
        if isinstance(v, str):
            return v.lower()
        return v
