"""Pydantic settings for ``arandu judge-answers`` (env prefix ``ARANDU_JUDGE_ANSWERS_``)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from arandu.shared.llm_settings import LLMSettings


class JudgeAnswersSettings(LLMSettings):
    """LLM + threshold settings for the answer judge (spec §12).

    Inherits the canonical LLM fields from :class:`LLMSettings`. The judge LLM
    is kept **separate** from the Answerer (a different model) so its judgments
    are not self-validated. Note that temperature does *not* mitigate
    self-preference: that bias is driven by familiarity / low perplexity, not
    by sampling settings (Wataoka et al. 2024), and is addressed by
    reference-grounding, which every answer criterion already has. Temperature
    is therefore set low (0.1) for reproducible single-shot scoring, matching
    the rest of the judge pipeline.

    Attributes:
        temperature: Sampling temperature. Default 0.1 for reproducible
            single-shot pointwise scoring (Wei et al. 2024).
        abstention_disagreement_audit: If True (default), emit
            ``abstention_audit.jsonl`` for items where the answerer's
            structured ``abstained`` flag disagrees with the abstention
            judge's verdict (the §6.4 disagreement signal).
    """

    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    abstention_disagreement_audit: bool = Field(default=True)

    model_config = SettingsConfigDict(env_prefix="ARANDU_JUDGE_ANSWERS_")
