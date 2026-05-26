"""Per-arm settings for ``arandu retrieve``.

Each retriever arm exposes a distinct set of knobs (chunker view, k-hop
radius, postings cap, LLM/embedder for atlas-rag's NER + edge filter,
…). Rather than overload one CLI command with a union of all knobs,
each arm has its own :class:`BaseSettings` subclass (env prefix
``ARANDU_<ARM>_``). The CLI instantiates only the settings classes for
arms it was asked to run.

Atlas-rag is the only arm that hits the network at retrieve time (NER
step in front of PPR). It needs an :class:`LLMClient` and a sentence
encoder; configuration lives in :class:`AtlasRagRetrieveSettings` plus
the existing :class:`arandu.shared.embeddings.EmbedderSettings`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from typing import Self


_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

ArmName = Literal["bm25", "atlas_rag", "khop_passage", "khop_triple", "null"]

# Stable ordering — drives the CLI's default arm set + the iteration
# order in the batch runner. Keeps benchmark output reproducible.
#
# ``atlas_rag`` is recognized but not yet wired in this PR: the factory
# routes it to a clear "deferred to a follow-up PR" error. Including it
# in the catalog so ``--arm atlas_rag`` produces the helpful error
# message rather than an unknown-arm rejection at the batch-runner
# validation layer. The CLI's ``_DEFAULT_ARMS`` filter excludes it.
ALL_ARMS: tuple[ArmName, ...] = (
    "bm25",
    "atlas_rag",
    "khop_passage",
    "khop_triple",
    "null",
)


class Bm25RetrieveSettings(BaseSettings):
    """Knobs for the BM25 arm.

    Attributes:
        chunker_id: ChunkSet view to query against. Must match a chunker
            id present in ``results/<id>/chunk/outputs/<chunker_id>/``
            (the chunking stage emits one directory per view it ran).
            Defaults to ``"cep_4k"`` — the project's primary view, used
            by the CEP QA generator.

    """

    chunker_id: str = Field(
        default="cep_4k",
        description="Chunker view to retrieve over (must exist under chunk/outputs/).",
    )

    model_config = SettingsConfigDict(env_prefix="ARANDU_BM25_", extra="ignore")


class AtlasRagRetrieveSettings(BaseSettings):
    """Knobs for the atlas-rag (HippoRAG-style) arm.

    atlas-rag uses an LLM at retrieve time for the NER step that
    identifies entities in the query before personalized PageRank
    seeding. The provider + model + base_url are read from
    ``ARANDU_ATLAS_RAG_*`` env vars; the sentence encoder is read
    separately from ``ARANDU_EMBEDDER_*`` (the same vars the
    ``arandu kg-build-retriever-index`` command uses, so the encoder
    that built the precompute matches the one used to embed queries).

    Attributes:
        provider: LLM provider for the NER step. ``"openai"``,
            ``"ollama"``, or ``"custom"``. Defaults to ``"openai"`` —
            the Gemini-compatible endpoint is the project's cloud path
            (set ``base_url`` to the Gemini URL when using it).
        model_id: Model identifier passed to the LLMClient (e.g.
            ``"gemini-2.5-flash"`` on Gemini, ``"qwen3:14b"`` on Ollama).
        api_key_env: Name of the env var holding the LLM API key. The
            CLI reads this env var when constructing the LLMClient.
        base_url: Optional explicit base URL. Required for ``custom``
            provider; ignored for ``openai``/``ollama`` defaults.
        keyword: atlas-rag's filename pattern. Defaults to project
            convention ``"transcriptions.json"`` (must match the value
            used at index-build time).
        include_events: Whether event nodes participate in the embedded
            node set. Must match the value used at index-build time —
            mismatches are caught by atlas-rag's manifest validator.
        include_concept: Whether concept nodes participate. Same
            mismatch-detection contract as ``include_events``.
    """

    provider: str = Field(
        default="openai",
        description="LLM provider for the NER step: openai, ollama, custom.",
    )
    model_id: str = Field(
        default="gemini-2.5-flash",
        description="Model identifier (e.g. gemini-2.5-flash, qwen3:14b).",
    )
    api_key_env: str = Field(
        default="GEMINI_API_KEY",
        description="Env var name holding the LLM API key.",
    )
    base_url: str | None = Field(
        default=None,
        description=(
            "Base URL for OpenAI-compatible endpoints. When unset, defaults "
            "are applied per-provider in the post-init validator: openai "
            "defaults to Gemini's compatibility URL (the project's primary "
            "cloud path); ollama leaves it None so LLMClient picks its own "
            "localhost default; custom requires an explicit value."
        ),
    )
    keyword: str = Field(
        default="transcriptions.json",
        description="atlas-rag filename pattern (must match index build).",
    )
    include_events: bool = Field(
        default=True,
        description="Include event nodes in the embedded set.",
    )
    include_concept: bool = Field(
        default=True,
        description="Include concept nodes in the embedded set.",
    )

    model_config = SettingsConfigDict(env_prefix="ARANDU_ATLAS_RAG_", extra="ignore")

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        """Normalize provider to lowercase so env-var case doesn't break dispatch."""
        if isinstance(v, str):
            return v.lower()
        return v

    @model_validator(mode="after")
    def _default_base_url_per_provider(self) -> Self:
        """Apply per-provider base_url defaults when the user hasn't set one.

        Without this, the field-level default of ``None`` would leave Gemini
        users with no base URL (LLMClient's PROVIDER_URLS[openai] is also
        None — OpenAI proper). Setting the project's primary cloud path
        here keeps the "just works" UX without baking the URL into a default
        that would override the ollama path's localhost endpoint.
        """
        if self.base_url is not None:
            return self
        if self.provider == "openai":
            object.__setattr__(self, "base_url", _GEMINI_OPENAI_BASE_URL)
        # ollama: leave None — LLMClient will use http://localhost:11434/v1.
        # custom: leave None — the factory will surface a clear error if
        # LLMClient can't resolve a URL.
        return self


class KHopRetrieveSettings(BaseSettings):
    """Knobs shared by the two k-hop arms (passage + triple).

    Both arms operate on the same atlas-rag KG and share the same
    entity-link + k-hop ego graph machinery. They differ only in their
    output unit (passages vs linearized triples), so the configuration
    surface is identical.

    Attributes:
        k_hop: Ego-graph radius around entity-linked seeds. Higher →
            more recall, slower, more dilution. The 2026-05-23 calibration
            against ``test-kg-04`` runs at ``k_hop=2`` for seconds-per-query
            wall time.
        max_postings: IDF-style cap on per-token entity-link expansion.
            Tokens whose inverted-index posting list exceeds this size
            are dropped from the entity link (e.g. ``"enchente"`` in a
            flood-themed corpus would otherwise link to thousands of
            entities and dominate the ego graph).
        keyword: atlas-rag's filename pattern for the graphml. Defaults
            to project convention ``"transcriptions.json"`` (see
            :mod:`arandu.kg.atlas_backend`).

    """

    k_hop: int = Field(
        default=2,
        ge=1,
        description="Ego-graph radius around entity-linked seeds.",
    )
    max_postings: int = Field(
        default=200,
        ge=1,
        description="IDF-style cap on per-token entity-link expansion.",
    )
    keyword: str = Field(
        default="transcriptions.json",
        description="atlas-rag filename pattern (rarely changed).",
    )

    model_config = SettingsConfigDict(env_prefix="ARANDU_KHOP_", extra="ignore")
