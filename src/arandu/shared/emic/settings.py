"""Settings for ``arandu emic-prepass`` (env prefix ``ARANDU_EMIC_PREPASS_``)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from arandu.shared.llm_settings import LLMSettings


class EmicPrepassSettings(LLMSettings):
    """LLM settings for the ordinal emic-validity pre-pass (spec §5).

    A thin subclass of :class:`LLMSettings` (same pattern as the answerer /
    judge / non-answerable stages): inherits the canonical LLM fields and the
    provider normalizer, pins the ``ARANDU_EMIC_PREPASS_`` env prefix, and
    overrides only the two defaults the pre-pass deliberately changes. The
    score is a sampling aid, **not** ground truth (the human annotators are the
    reference), so a modest model is fine and the pre-pass can run a cheaper
    model than the judge via ``ARANDU_EMIC_PREPASS_MODEL_ID``.

    Attributes:
        temperature: Sampling temperature. Default 0.1 — the emic judgment is
            structural, not creative (spec §4.2 principle 8). Still
            env-overridable via ``ARANDU_EMIC_PREPASS_TEMPERATURE``.
        language: Prompt language. Narrowed to ``"pt"`` only — that is the only
            ``emic_validity`` prompt template that ships today.
    """

    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    language: Literal["pt"] = Field(default="pt")

    model_config = SettingsConfigDict(env_prefix="ARANDU_EMIC_PREPASS_")
