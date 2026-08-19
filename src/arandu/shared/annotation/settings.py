"""Settings for the Label Studio instance (env prefix ``ARANDU_LABEL_STUDIO_``)."""

from __future__ import annotations

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LabelStudioSettings(BaseSettings):
    """Connection settings for the annotation platform.

    The token is required and has no default: a missing credential must fail
    loudly at startup rather than produce a confusing 401 mid-push. It is read
    from the environment or from ``.env``, matching every other settings class
    in this codebase (``ResultsConfig``, ``KGConfig``, the QA and transcription
    configs), so a documented ``.env`` entry is not silently ignored.

    ``min_length=1`` makes an empty value fail the same way an absent one does.
    ``.env.example`` ships ``ARANDU_LABEL_STUDIO_TOKEN=`` with no value, since
    the credential is only needed by the two commands that talk to the instance;
    without the constraint, the ordinary copy-the-example flow validates and
    sends ``Authorization: Token `` instead of printing the CLI's "Label Studio
    is not configured" message. The class is only ever constructed inside those
    two commands (and in ``pull`` only on the network path), so this constrains
    nothing for a checkout that never annotates.

    The token value never reaches an artifact under ``results/``. Two mechanisms
    keep it out, because this stage writes the study's most sensitive output:
    ``SecretStr`` masks it in every repr and ``model_dump()`` (so it cannot
    reach a ``ConfigSnapshot``'s ``config_values``), and
    :func:`~arandu.shared.schemas.is_secret_env_name` redacts the raw
    ``ARANDU_LABEL_STUDIO_TOKEN`` environment variable out of the environment
    snapshot every stage writes into its ``run_metadata.json``.

    ``token`` is a :class:`~pydantic.SecretStr`, not a plain ``str``: this is
    the first settings class in this codebase to hold a live secret value
    (``LLMSettings`` only stores an env-var *name*, ``api_key_env``, never the
    key itself). ``SecretStr`` keeps the token out of Pydantic's default repr
    and out of ``model_dump()``, so a future stray ``logger.info(settings)`` or
    an unhandled exception that renders the object cannot print it. Callers
    that need the raw value call ``token.get_secret_value()`` at the point of
    use (see :func:`arandu.shared.annotation.client.build_client_from_settings`).

    Attributes:
        url: Base URL of the instance, without a trailing slash.
        token: Label Studio API token (Account and Settings, Access Token).
        timeout: Per-request timeout in seconds. Generous by default: a 120-task
            import is one request.
    """

    url: str = Field(..., description="Base URL of the Label Studio instance")
    token: SecretStr = Field(..., min_length=1, description="Label Studio API token")
    timeout: float = Field(default=60.0, gt=0, description="Per-request timeout in seconds")

    model_config = SettingsConfigDict(
        env_prefix="ARANDU_LABEL_STUDIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        """Normalize the base URL so path joins never double the separator."""
        return value.rstrip("/")
