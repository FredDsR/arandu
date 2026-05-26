"""Unified ``chunk_id → Chunk`` map for the answer-judge batch.

Three retriever families produce three chunk_id namespaces (per the
``arandu retrieve`` and ``arandu answer`` design):

- **BM25** chunk_ids resolve directly to :class:`Chunk` objects in the
  on-disk ChunkSets (``chunk/outputs/<chunker_id>/*.json``).
- **atlas-rag / khop_passage** chunk_ids are atlas-rag-synthesised
  ``"<file_id>:<idx>"`` that join to the
  :class:`PassageOffsetSidecar` from PR #100.
- **khop_triple** chunk_ids are ``"triple:<sha>"`` synthetic ids with
  no source-text offsets at all; they simply don't appear in this map
  (offset_coverage drops them).

The deterministic ``offset_coverage`` criterion needs :class:`Chunk`
objects to compute character-range overlap with the gold chunk; this
helper produces that unified view.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from arandu.kg.passage_offsets import PassageOffsetSidecar
from arandu.shared.chunking.schemas import Chunk, ChunkSet

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def build_retrieved_chunks_map(
    *,
    chunk_outputs_root: Path,
    passage_offsets_path: Path | None,
) -> dict[str, Chunk]:
    """Build ``chunk_id → Chunk`` for every BM25 + atlas-rag chunk in a run.

    Args:
        chunk_outputs_root: ``results/<id>/chunk/outputs/``. Each
            subdirectory is a chunker view; flattened recursively.
        passage_offsets_path: ``results/<id>/kg/outputs/passage_offsets.json``.
            ``None`` (or non-existent) is tolerated for runs without
            atlas-rag.

    Returns:
        A flat ``chunk_id → Chunk`` dict. Triple-arm chunk_ids
        (``"triple:<sha>"``) deliberately won't appear — they have no
        source-text offsets.
    """
    out: dict[str, Chunk] = {}
    out.update(_bm25_chunks(chunk_outputs_root))
    if passage_offsets_path is not None and passage_offsets_path.exists():
        out.update(_atlas_chunks(passage_offsets_path))
    return out


def _bm25_chunks(chunk_outputs_root: Path) -> dict[str, Chunk]:
    """Flatten every ChunkSet under ``chunk_outputs_root`` into ``chunk_id → Chunk``."""
    out: dict[str, Chunk] = {}
    if not chunk_outputs_root.exists():
        return out
    for view_dir in chunk_outputs_root.iterdir():
        if not view_dir.is_dir():
            continue
        for path in sorted(view_dir.glob("*.json")):
            try:
                chunk_set = ChunkSet.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValidationError) as exc:
                logger.warning("Skipping unreadable ChunkSet %s: %s", path, exc)
                continue
            for chunks in chunk_set.views.values():
                for chunk in chunks:
                    out[chunk.chunk_id] = chunk
    return out


def _atlas_chunks(passage_offsets_path: Path) -> dict[str, Chunk]:
    """Convert each :class:`PassageOffset` into a :class:`Chunk` keyed by passage_id.

    atlas-rag's synthetic passage_id (``"<file_id>:<idx>"``) IS the
    retrieve-side chunk_id (per PR #100's join contract), so the key
    here joins directly to ``RetrievedPassage.chunk_id`` at judge time.
    """
    try:
        sidecar = PassageOffsetSidecar.load(passage_offsets_path)
    except (OSError, ValidationError) as exc:
        logger.warning("Skipping unreadable passage_offsets %s: %s", passage_offsets_path, exc)
        return {}
    out: dict[str, Chunk] = {}
    for offset in sidecar.offsets:
        out[offset.passage_id] = Chunk(
            chunk_id=offset.passage_id,
            source_file_id=offset.source_file_id,
            start_char=offset.start_char,
            end_char=offset.end_char,
            chunker_id=offset.chunker_id,
        )
    return out
