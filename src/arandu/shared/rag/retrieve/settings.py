"""Per-arm settings for ``arandu retrieve``.

Each retriever arm exposes a distinct set of knobs (chunker view, k-hop
radius, postings cap, …). Rather than overload one CLI command with a
union of all knobs, each arm has its own :class:`BaseSettings` subclass
(env prefix ``ARANDU_<ARM>_``). The CLI instantiates only the settings
classes for arms it was asked to run.

Atlas-rag's settings live in a follow-up PR — that arm needs an LLM
client + sentence encoder at retrieve time and is intentionally kept
out of this PR's scope.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
