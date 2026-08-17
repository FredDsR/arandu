"""Settings for the Label Studio instance (env prefix ``ARANDU_LABEL_STUDIO_``)."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LabelStudioSettings(BaseSettings):
    """Connection settings for the annotation platform.

    The token is required and has no default: a missing credential must fail
    loudly at startup rather than produce a confusing 401 mid-push. It is read
    from the environment and never written to any artifact under ``results/``,
    because the labels directory is the study's most sensitive output.

    Attributes:
        url: Base URL of the instance, without a trailing slash.
        token: Label Studio API token (Account and Settings, Access Token).
        timeout: Per-request timeout in seconds. Generous by default: a 120-task
            import is one request.
    """

    url: str = Field(..., description="Base URL of the Label Studio instance")
    token: str = Field(..., description="Label Studio API token")
    timeout: float = Field(default=60.0, gt=0, description="Per-request timeout in seconds")

    model_config = SettingsConfigDict(env_prefix="ARANDU_LABEL_STUDIO_", extra="ignore")

    @field_validator("url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        """Normalize the base URL so path joins never double the separator."""
        return value.rstrip("/")
