"""Pydantic settings for ``arandu judge-answers`` (env prefix ``ARANDU_JUDGE_ANSWERS_``)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from arandu.shared.llm_settings import LLMSettings


class JudgeAnswersSettings(LLMSettings):
    """LLM + threshold settings for the answer judge (spec §12).

    Inherits the canonical LLM fields from :class:`LLMSettings`. The judge LLM
    is **separate** from the Answerer — using a different model (or at least a
    different temperature) reduces the chance that the Answerer's own
    systematic biases are self-validated, which is why the default temperature
    is bumped to 0.3.

    Attributes:
        temperature: Sampling temperature. Default 0.3 — slightly higher than
            the answerer's 0.2 to encourage less-anchored judgment.
        abstention_disagreement_audit: If True (default), emit
            ``abstention_audit.jsonl`` for items where the answerer's
            structured ``abstained`` flag disagrees with the abstention
            judge's verdict (the §6.4 disagreement signal).
    """

    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    abstention_disagreement_audit: bool = Field(default=True)

    model_config = SettingsConfigDict(env_prefix="ARANDU_JUDGE_ANSWERS_")
