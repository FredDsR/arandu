"""Settings for ``arandu emic-judge`` (env prefix ``ARANDU_EMIC_JUDGE_``)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from arandu.shared.llm_settings import LLMSettings


class EmicJudgeSettings(LLMSettings):
    """LLM settings for the ordinal emic-validity judge (spec §5).

    A thin subclass of :class:`LLMSettings` (same pattern as the answerer /
    judge / non-answerable stages): inherits the canonical LLM fields and the
    provider normalizer, pins the ``ARANDU_EMIC_JUDGE_`` env prefix, and
    overrides only the two defaults the emic judgment deliberately changes.

    Model choice is a **methodological** decision here, not a budget one: the
    scores this stage produces are the measurement the study reports, and the
    human annotation validates them (it does not replace them). Changing
    ``ARANDU_EMIC_JUDGE_MODEL_ID`` changes the instrument under test, so a run
    used for the agreement study must pin the same model as the one the thesis
    describes.

    Attributes:
        temperature: Sampling temperature. Default 0.1: the emic judgment is
            structural, not creative (spec §4.2 principle 8). Still
            env-overridable via ``ARANDU_EMIC_JUDGE_TEMPERATURE``.
        language: Prompt language. Narrowed to ``"pt"`` only (that is the only
            ``emic_validity`` prompt template that ships today).
    """

    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    language: Literal["pt"] = Field(default="pt")

    model_config = SettingsConfigDict(env_prefix="ARANDU_EMIC_JUDGE_")
