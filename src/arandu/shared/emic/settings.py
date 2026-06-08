"""Settings for ``arandu emic-prepass`` (env prefix ``ARANDU_EMIC_PREPASS_``).

The emic pre-pass needs nothing beyond the canonical LLM configuration, so it
reuses :class:`~arandu.shared.llm_client.LLMSettings` directly rather than
declaring a bespoke settings class. This module only pins the stage's env
prefix and its one deliberate default (a low, structural-not-creative
temperature, spec §4.2 principle 8).
"""

from __future__ import annotations

from arandu.shared.llm_client import LLMSettings

EMIC_ENV_PREFIX = "ARANDU_EMIC_PREPASS_"


def default_emic_settings() -> LLMSettings:
    """Return the default LLM settings for the emic pre-pass.

    Reads ``ARANDU_EMIC_PREPASS_*`` (provider/model_id/api_key_env/base_url
    remain env-overridable so the pre-pass can run a cheaper model than the
    judge). Temperature is pinned to 0.1: the emic judgment is structural, not
    creative (spec §4.2 principle 8). The score is a sampling aid, not ground
    truth (the human annotators are the reference).
    """
    return LLMSettings(_env_prefix=EMIC_ENV_PREFIX, temperature=0.1)
