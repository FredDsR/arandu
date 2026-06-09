"""Pydantic settings for the answerer (env prefix ``ARANDU_ANSWERER_``)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from arandu.shared.llm_settings import LLMSettings


class AnswererSettings(LLMSettings):
    """LLM + budget settings for the Answerer (spec §5.7).

    Inherits the canonical LLM connection/sampling fields from
    :class:`LLMSettings`; adds the answerer's token-budget knobs. Held
    constant across retrieval arms in a single run; this is what isolates
    retrieval-quality from generation-quality in the cross-arm comparison
    (spec §5.1).

    Attributes:
        max_tokens: Max tokens in the answerer's response. Defaults to 8192:
            although the free-text answer is short, the answerer emits
            *structured* output, and reasoning models (gemini-2.5-flash,
            qwen3:14b) spend thinking tokens against this budget. A small
            ceiling (the old 1024) truncated the JSON mid-string, so the
            answerer fell back to ``abstained`` and polluted the answerable
            arms with spurious abstentions (dry-run 2026-06-08). Matches the
            judge/CEP reasoning-model headroom (see
            :data:`arandu.shared.judge.criterion.DEFAULT_MAX_TOKENS`).
        top_k: Maximum passages to consider per question. Defaults to 10;
            the actual count may be smaller after token-budget packing.
        max_context_tokens: Total token budget (passages + prompt + answer)
            the answerer packs into one call. Default 16384.
            :func:`pack_passages` derives the passage budget as
            ``max_context_tokens - prompt_overhead_tokens - max_tokens``, so it
            must exceed the reasoning-model ``max_tokens`` (8192) with room left
            for passages — an 8192 ceiling here leaves a negative budget. This
            is an artificial benchmark cap held constant across arms, not the
            model's real context window (gemini-2.5-flash / qwen3:14b are far
            larger).
        prompt_overhead_tokens: Reserved budget for the rendered prompt
            template (everything except the passage list). The spec's default
            of 350 is an empirical estimate; tune downward if larger questions
            blow the budget.
    """

    max_tokens: int = Field(default=8192, gt=0)
    top_k: int = Field(default=10, ge=1)
    max_context_tokens: int = Field(default=16384, gt=0)
    prompt_overhead_tokens: int = Field(default=350, ge=0)

    model_config = SettingsConfigDict(env_prefix="ARANDU_ANSWERER_")
