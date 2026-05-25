"""Embedder settings + factory dispatch.

Two providers supported today:

- ``gemini``: cloud, OpenAI-compatible endpoint, sized for thesis runs.
  See :class:`arandu.shared.embeddings.gemini.GeminiEmbedder`.
- ``sentence_transformers``: local, GPU-backed; used on PCAD / when
  cloud quotas are tight. atlas-rag's
  :class:`SentenceTransformerEmbeddingModel` is used as-is.

The choice is exposed through :class:`EmbedderSettings` (env prefix
``ARANDU_EMBEDDER_``) and resolved by :func:`build_embedder`.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from arandu.shared.embeddings.gemini import GeminiEmbedder

EmbedderProvider = Literal["gemini", "sentence_transformers"]


_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class EmbedderSettings(BaseSettings):
    """Provider + model selection for atlas-rag's sentence encoder.

    Attributes:
        provider: Embedding backend. Defaults to ``"gemini"`` because the
            thesis runs are cloud-driven; switch to
            ``"sentence_transformers"`` for PCAD or other GPU-local paths.
        model: Provider-specific model identifier. For Gemini, defaults to
            ``"gemini-embedding-001"`` (the 004 series was removed by
            Google in 2025-Q1). For sentence-transformers, any
            ``sentence_transformers``-resolvable name works
            (e.g. ``"intfloat/multilingual-e5-large-instruct"``).
        api_key_env: Name of the env var holding the Gemini API key.
            Defaults to ``GEMINI_API_KEY`` — matching the smoke script's
            convention. Ignored for sentence-transformers.
    """

    provider: EmbedderProvider = Field(
        default="gemini",
        description="Embedding backend: 'gemini' (cloud) or 'sentence_transformers' (local).",
    )
    model: str = Field(
        default="gemini-embedding-001",
        description="Model identifier passed to the chosen provider.",
    )
    api_key_env: str = Field(
        default="GEMINI_API_KEY",
        description="Env var name holding the Gemini API key (ignored for local providers).",
    )

    model_config = SettingsConfigDict(env_prefix="ARANDU_EMBEDDER_", extra="ignore")


def build_embedder(settings: EmbedderSettings) -> Any:
    """Return a duck-typed embedder for the configured provider.

    Both return types satisfy atlas-rag's encoder contract
    (``encode(list[str], normalize_embeddings=bool) -> ndarray``); the
    return is typed as ``Any`` because atlas-rag's
    :class:`BaseEmbeddingModel` is a heavy class hierarchy we don't want
    to import at module-load time.

    Args:
        settings: Resolved settings (typically built via
            ``EmbedderSettings()`` so env vars are read).

    Returns:
        An embedder satisfying atlas-rag's encoder contract.

    Raises:
        RuntimeError: If ``provider`` is ``"gemini"`` and the API key
            env var (default ``GEMINI_API_KEY``) is unset.
        ValueError: If ``provider`` is not a recognized value (defensive;
            ``EmbedderProvider`` is a Literal, but env-var paths can
            still feed an arbitrary string).
    """
    if settings.provider == "gemini":
        api_key = os.environ.get(settings.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Gemini embedder requested but {settings.api_key_env} is unset. "
                f"Set it or switch ARANDU_EMBEDDER_PROVIDER to 'sentence_transformers'."
            )
        # Import the OpenAI client lazily so the sentence-transformers path
        # works in envs without `openai` installed.
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=_GEMINI_OPENAI_BASE_URL)
        return GeminiEmbedder(client=client, model=settings.model)

    if settings.provider == "sentence_transformers":
        # Defer atlas-rag import; only the local-GPU path needs it.
        from atlas_rag.vectorstore.embedding_model import SentenceTransformerEmbeddingModel

        return SentenceTransformerEmbeddingModel(settings.model)

    raise ValueError(
        f"Unknown embedder provider {settings.provider!r}. "
        f"Valid values: 'gemini', 'sentence_transformers'."
    )
