"""Construct retriever instances from a pipeline id + arm-specific settings.

The CLI passes a flat ``(arm, pipeline_id, settings, rebuild_index)`` to
:func:`build_retriever`; this module hides the per-arm wiring (chunk
loading + BM25 index build, KG path resolution, LLM client + sentence
encoder for atlas-rag, …).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from pydantic import ValidationError

from arandu.shared.chunking.resolver import ChunkResolver
from arandu.shared.chunking.schemas import Chunk, ChunkSet
from arandu.shared.config import ResultsConfig
from arandu.shared.embeddings import EmbedderSettings, build_embedder
from arandu.shared.llm_client import LLMClient, LLMProvider
from arandu.shared.rag.retrieve.settings import (
    AtlasRagRetrieveSettings,
    Bm25RetrieveSettings,
    KHopRetrieveSettings,
)
from arandu.shared.rag.retrievers.atlas_rag import (
    MANIFEST_FILENAME,
    PRECOMPUTE_DIR_NAME,
    AtlasRagRetriever,
)
from arandu.shared.rag.retrievers.bm25 import BM25Retriever
from arandu.shared.rag.retrievers.khop_subgraph import KHopSubgraphRetriever
from arandu.shared.rag.retrievers.khop_triple import KHopTripleRetriever
from arandu.shared.rag.retrievers.null import NullRetriever
from arandu.shared.schemas import EnrichedRecord

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.shared.rag.protocol import Retriever
    from arandu.shared.rag.retrieve.settings import ArmName

logger = logging.getLogger(__name__)


_BM25_INDEX_FILENAME = "bm25.pkl"


RetrieveSettings = Bm25RetrieveSettings | KHopRetrieveSettings | AtlasRagRetrieveSettings


def build_retriever(
    arm: ArmName,
    *,
    pipeline_id: str,
    settings: RetrieveSettings | None = None,
    embedder_settings: EmbedderSettings | None = None,
    base_dir: Path | None = None,
    rebuild_index: bool = False,
) -> Retriever:
    """Construct the retriever for ``arm`` against ``pipeline_id``'s artifacts.

    Args:
        arm: One of ``"bm25"``, ``"atlas_rag"``, ``"khop_passage"``,
            ``"khop_triple"``, ``"null"``.
        pipeline_id: The run identifier; all paths resolve relative to
            ``<base_dir>/<pipeline_id>/``.
        settings: Arm-specific settings instance. For ``null``, ignored.
            For ``bm25``, must be :class:`Bm25RetrieveSettings`; for
            k-hop arms, :class:`KHopRetrieveSettings`; for atlas-rag,
            :class:`AtlasRagRetrieveSettings`. Defaults are instantiated
            when omitted.
        embedder_settings: Sentence-encoder choice (atlas-rag arm only,
            ignored otherwise). Defaults to :class:`EmbedderSettings()`,
            which reads ``ARANDU_EMBEDDER_*`` env vars — the same vars
            ``arandu kg-build-retriever-index`` consumes, so the encoder
            that built the precompute matches the one used to embed
            queries at retrieve time.
        base_dir: Project ``results/`` root. Defaults to
            ``ResultsConfig().base_dir``.
        rebuild_index: Force-rebuild arm-side indexes (BM25 pickle only;
            no-op for k-hop and null, which build their state in-memory
            from the KG/graphml at construction; for atlas-rag the
            precompute is the responsibility of
            ``arandu kg-build-retriever-index``).

    Returns:
        A retriever instance satisfying :class:`Retriever`.

    Raises:
        FileNotFoundError: If required artifacts (chunks, KG outputs,
            atlas-rag precompute) are missing for ``pipeline_id``.
        RuntimeError: From :func:`build_embedder` (atlas-rag arm) if the
            embedder API key is unset, or from the atlas-rag LLM
            construction if its API key env var is unset.
        ValueError: For unknown arm names or invalid arm/settings combos.

    """
    base = base_dir if base_dir is not None else ResultsConfig().base_dir

    if arm == "null":
        return NullRetriever()

    if arm == "bm25":
        bm25_settings = (
            settings if isinstance(settings, Bm25RetrieveSettings) else Bm25RetrieveSettings()
        )
        return _build_bm25(
            pipeline_id=pipeline_id,
            base_dir=base,
            chunker_id=bm25_settings.chunker_id,
            rebuild_index=rebuild_index,
        )

    if arm in ("khop_passage", "khop_triple"):
        khop_settings = (
            settings if isinstance(settings, KHopRetrieveSettings) else KHopRetrieveSettings()
        )
        kg_outputs_dir = base / pipeline_id / "kg" / "outputs" / "atlas_output"
        if not kg_outputs_dir.exists():
            raise FileNotFoundError(
                f"atlas-rag KG outputs not found for pipeline_id {pipeline_id!r}: "
                f"{kg_outputs_dir}. Run `arandu build-kg --id {pipeline_id}` first."
            )
        retriever_cls = KHopSubgraphRetriever if arm == "khop_passage" else KHopTripleRetriever
        return retriever_cls(
            kg_outputs_dir=kg_outputs_dir,
            keyword=khop_settings.keyword,
            k_hop=khop_settings.k_hop,
            max_postings=khop_settings.max_postings,
        )

    if arm == "atlas_rag":
        atlas_settings = (
            settings
            if isinstance(settings, AtlasRagRetrieveSettings)
            else AtlasRagRetrieveSettings()
        )
        emb_settings = embedder_settings if embedder_settings is not None else EmbedderSettings()
        return _build_atlas_rag(
            pipeline_id=pipeline_id,
            base_dir=base,
            settings=atlas_settings,
            embedder_settings=emb_settings,
        )

    raise ValueError(f"Unknown arm {arm!r}.")


def _build_bm25(
    *,
    pipeline_id: str,
    base_dir: Path,
    chunker_id: str,
    rebuild_index: bool,
) -> BM25Retriever:
    """Load (or build) the BM25 index for ``chunker_id`` and return the retriever.

    The pickle lives under
    ``<base_dir>/<pipeline_id>/retrieve/indexes/bm25_<chunker_id>/`` —
    co-located with the retrieve stage's outputs so the index follows
    the same run-id lifecycle as the rest of the benchmark artifacts.
    """
    chunk_dir = base_dir / pipeline_id / "chunk" / "outputs" / chunker_id
    if not chunk_dir.exists():
        raise FileNotFoundError(
            f"Chunks not found for chunker_id {chunker_id!r} under {chunk_dir}. "
            f"Run `arandu chunk --id {pipeline_id} --view {chunker_id}` first."
        )

    index_dir = base_dir / pipeline_id / "retrieve" / "indexes" / f"bm25_{chunker_id}"
    if rebuild_index or not (index_dir / _BM25_INDEX_FILENAME).exists():
        chunks = _load_chunks(chunk_dir, chunker_id)
        if not chunks:
            raise FileNotFoundError(
                f"No chunks found in {chunk_dir} for chunker_id {chunker_id!r}. "
                f"The chunking stage produced no output for this view."
            )
        resolver = _build_chunk_resolver(pipeline_id=pipeline_id, base_dir=base_dir)
        BM25Retriever.build_index(
            chunks=chunks,
            resolver=resolver,
            index_dir=index_dir,
            chunker_id=chunker_id,
        )

    return BM25Retriever(index_dir=index_dir, chunker_id=chunker_id)


def _load_chunks(chunk_dir: Path, chunker_id: str) -> list[Chunk]:
    """Flatten every ``ChunkSet`` under ``chunk_dir`` into a single chunk list.

    Each on-disk file is a single-view ``ChunkSet`` (per the chunking
    batch convention in PR #99: one file per (chunker, source) pair).
    """
    chunks: list[Chunk] = []
    for path in sorted(chunk_dir.glob("*.json")):
        try:
            chunk_set = ChunkSet.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            logger.warning("Skipping unreadable ChunkSet %s: %s", path, exc)
            continue
        if chunker_id not in chunk_set.views:
            logger.warning(
                "ChunkSet %s has no view %r (has: %s); skipping.",
                path,
                chunker_id,
                sorted(chunk_set.views),
            )
            continue
        chunks.extend(chunk_set.views[chunker_id])
    return chunks


def _build_chunk_resolver(*, pipeline_id: str, base_dir: Path) -> ChunkResolver:
    """Construct a :class:`ChunkResolver` reading source text from this run.

    Source text lives at ``<base_dir>/<pipeline_id>/transcription/outputs/
    <file_id>.json`` as an :class:`EnrichedRecord`. The loader reads it
    lazily; ``ChunkResolver`` caches loaded texts in its LRU.
    """
    transcription_dir = base_dir / pipeline_id / "transcription" / "outputs"
    if not transcription_dir.exists():
        raise FileNotFoundError(
            f"Transcription outputs not found at {transcription_dir}. "
            f"BM25 index building needs source text to resolve chunks."
        )

    def _load_text(file_id: str) -> str:
        path = transcription_dir / f"{file_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Transcription not found for file_id {file_id!r}: {path}")
        record = EnrichedRecord.model_validate_json(path.read_text(encoding="utf-8"))
        return record.transcription_text

    return ChunkResolver(text_loader=_load_text)


def _build_atlas_rag(
    *,
    pipeline_id: str,
    base_dir: Path,
    settings: AtlasRagRetrieveSettings,
    embedder_settings: EmbedderSettings,
) -> AtlasRagRetriever:
    """Construct :class:`AtlasRagRetriever` for ``pipeline_id``.

    Verifies the precompute exists (built separately by
    ``arandu kg-build-retriever-index``), constructs the LLM client + the
    sentence encoder, and dispatches to the retriever's constructor —
    which then runs its own integrity validation against the manifest
    (graphml sha + per-pickle sha + flag-mismatch checks).
    """
    atlas_output_dir = base_dir / pipeline_id / "kg" / "outputs" / "atlas_output"
    if not atlas_output_dir.exists():
        raise FileNotFoundError(
            f"atlas-rag KG outputs not found for pipeline_id {pipeline_id!r}: "
            f"{atlas_output_dir}. Run `arandu build-kg --id {pipeline_id}` first."
        )

    precompute_manifest = atlas_output_dir / PRECOMPUTE_DIR_NAME / MANIFEST_FILENAME
    if not precompute_manifest.exists():
        raise FileNotFoundError(
            f"atlas-rag precompute manifest not found at {precompute_manifest}. "
            f"Run `arandu kg-build-retriever-index --id {pipeline_id}` first "
            f"(this is the LLM-spending step that builds the embeddings index "
            f"the atlas-rag retriever reads at retrieve time)."
        )

    api_key = os.environ.get(settings.api_key_env)
    if not api_key and settings.provider != "ollama":
        # Ollama lets a bogus key through; cloud providers don't.
        raise RuntimeError(
            f"atlas_rag arm requested but {settings.api_key_env} is unset. "
            f"Set it or switch ARANDU_ATLAS_RAG_PROVIDER to 'ollama'."
        )

    llm_client = LLMClient(
        provider=LLMProvider(settings.provider),
        model_id=settings.model_id,
        api_key=api_key,
        base_url=settings.base_url,
    )
    encoder = build_embedder(embedder_settings)

    return AtlasRagRetriever(
        kg_outputs_dir=atlas_output_dir,
        llm_client=llm_client.client,
        llm_model_id=llm_client.model_id,
        sentence_encoder=encoder,
        sentence_encoder_model=embedder_settings.model,
        keyword=settings.keyword,
        include_events=settings.include_events,
        include_concept=settings.include_concept,
    )
