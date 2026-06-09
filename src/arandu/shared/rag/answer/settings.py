"""Pydantic settings for the answerer (env prefix ``ARANDU_ANSWERER_``)."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from arandu.shared.llm_settings import LLMSettings


class AnswererSettings(LLMSettings):
    """LLM + budget settings for the Answerer (spec §5.7).

    Inherits the canonical LLM connection/sampling fields from
    :class:`LLMSettings`; adds the answerer's token-budget knobs. Held
    constant across retrieval arms in a single run; this is what isolates
    retrieval-quality from generation-quality in the cross-arm comparison
    (spec §5.1).

    ``max_tokens`` is inherited unchanged from :class:`LLMSettings` (the shared
    :data:`~arandu.shared.llm_settings.REASONING_MODEL_MAX_TOKENS` headroom):
    although the answer is short, it is emitted as *structured* output and
    reasoning models spend thinking tokens against the budget, so a tight cap
    truncates the JSON and the answerer falls back to ``abstained`` (dry-run
    2026-06-08). The packing budget invariant tying it to ``max_context_tokens``
    is enforced by :meth:`_check_packing_budget`.

    Attributes:
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

    top_k: int = Field(default=10, ge=1)
    max_context_tokens: int = Field(default=16384, gt=0)
    prompt_overhead_tokens: int = Field(default=350, ge=0)

    model_config = SettingsConfigDict(env_prefix="ARANDU_ANSWERER_")

    @model_validator(mode="after")
    def _check_packing_budget(self) -> Self:
        """Fail fast if no token budget is left for passages.

        :func:`~arandu.shared.rag.answer.packer.pack_passages` derives the
        passage budget as ``max_context_tokens - prompt_overhead_tokens -
        max_tokens`` and raises if it is non-positive. Because all three fields
        are independently overridable (``ARANDU_ANSWERER_*``), a bad combination
        would otherwise only surface mid-run on the first question; catch it at
        construction instead.
        """
        budget = self.max_context_tokens - self.prompt_overhead_tokens - self.max_tokens
        if budget <= 0:
            raise ValueError(
                f"Answerer packing budget is non-positive: max_context_tokens "
                f"({self.max_context_tokens}) - prompt_overhead_tokens "
                f"({self.prompt_overhead_tokens}) - max_tokens ({self.max_tokens}) "
                f"= {budget}. Raise max_context_tokens or lower max_tokens / "
                f"prompt_overhead_tokens so pack_passages has room for passages."
            )
        return self
