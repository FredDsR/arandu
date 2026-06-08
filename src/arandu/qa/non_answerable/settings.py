"""Pydantic settings for non-answerable generation (env ``ARANDU_NONANSWERABLE_``)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from arandu.shared.llm_settings import LLMSettings


class NonAnswerableSettings(LLMSettings):
    """LLM + sampling settings for the non-answerable builder (spec §7, §12).

    Inherits the canonical LLM connection fields from :class:`LLMSettings`.
    The builder drives the LLM with its own per-attempt ``base_temperature``
    (escalated on retry), so the inherited ``temperature``/``max_tokens`` are
    not used here.

    Attributes:
        seeds_per_bloom: Target seeds per Bloom level. 100 -> ~400 total.
        rng_seed: Sampler seed for reproducibility.
        retry_max: Collision retries per seed before skipping it.
        base_temperature: Temperature for the first perturbation attempt;
            each retry adds 0.1 to diversify the sample.
    """

    seeds_per_bloom: int = Field(default=100, ge=1)
    rng_seed: int = Field(default=42)
    retry_max: int = Field(default=3, ge=1)
    base_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    model_config = SettingsConfigDict(env_prefix="ARANDU_NONANSWERABLE_")
