"""Pydantic settings for the answerer (env prefix ``ARANDU_ANSWERER_``)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from arandu.shared.llm_client import LLMSettings


class AnswererSettings(LLMSettings):
    """LLM + budget settings for the Answerer (spec §5.7).

    Inherits the canonical LLM connection/sampling fields from
    :class:`LLMSettings`; adds the answerer's token-budget knobs. Held
    constant across retrieval arms in a single run — this is what isolates
    retrieval-quality from generation-quality in the cross-arm comparison
    (spec §5.1).

    Attributes:
        max_tokens: Max tokens in the answerer's response. Overrides the
            base default down to 1024 (answers are short).
        top_k: Maximum passages to consider per question. Defaults to 10;
            the actual count may be smaller after token-budget packing.
        max_context_tokens: Total context window for the answerer. Default
            8192. Used by :func:`pack_passages` to compute the passage budget.
        prompt_overhead_tokens: Reserved budget for the rendered prompt
            template (everything except the passage list). The spec's default
            of 350 is an empirical estimate; tune downward if larger questions
            blow the budget.
    """

    max_tokens: int = Field(default=1024, gt=0)
    top_k: int = Field(default=10, ge=1)
    max_context_tokens: int = Field(default=8192, gt=0)
    prompt_overhead_tokens: int = Field(default=350, ge=0)

    model_config = SettingsConfigDict(env_prefix="ARANDU_ANSWERER_", extra="ignore")
