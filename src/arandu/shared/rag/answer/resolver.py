"""Build the ``chunk_id → text`` map an answerer batch needs.

Three retriever families produce three chunk_id namespaces:

- **BM25** emits chunks from a :class:`ChunkSet` view; the text is the
  source transcription sliced by ``(start_char, end_char)``.
- **atlas-rag / khop_passage** emit synthesised passage_ids of the form
  ``"<source_file_id>:<chunk_index>"`` that join to the
  :class:`PassageOffsetSidecar` from PR #100. The text is again the
  source transcription sliced by the sidecar's offsets.
- **khop_triple** emits ``"triple:<sha>"`` ids with the text already
  in ``RetrievedPassage.payload``; the packer handles those without
  consulting this map.

This helper unions the BM25 + atlas-rag/khop_passage resolutions into a
single dict the batch driver can pass to :func:`pack_passages`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arandu.kg.passage_offsets import PassageOffsetSidecar
from arandu.shared.chunking.schemas import ChunkSet
from arandu.shared.schemas import EnrichedRecord

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def build_passage_text_map(
    *,
    chunk_dirs: list[Path],
    passage_offsets_path: Path | None,
    transcription_dir: Path,
) -> dict[str, str]:
    """Build the ``chunk_id → text`` map for one run.

    Args:
        chunk_dirs: Per-chunker view directories under
            ``results/<id>/chunk/outputs/`` whose ChunkSets back the
            BM25 arm. Each directory holds ``<file_id>.json`` files
            with single-view ChunkSets (per PR #99's convention).
        passage_offsets_path: Path to
            ``results/<id>/kg/outputs/passage_offsets.json``. ``None``
            (or non-existent) is tolerated when only BM25 arms ran.
        transcription_dir: ``results/<id>/transcription/outputs/`` —
            holds the source :class:`EnrichedRecord` files whose
            ``transcription_text`` is sliced by both BM25 chunks and
            the atlas-rag sidecar offsets.

    Returns:
        A flat ``chunk_id → text`` dict. Unresolvable references
        (missing source transcription, missing sidecar entry, …) are
        omitted; downstream packing simply drops them.
    """
    text_by_file_id = _load_transcriptions(transcription_dir)

    out: dict[str, str] = {}
    for chunk_dir in chunk_dirs:
        if not chunk_dir.exists():
            logger.debug("Chunk dir absent: %s; skipping for BM25 resolution.", chunk_dir)
            continue
        out.update(_bm25_chunks_to_text(chunk_dir, text_by_file_id))

    if passage_offsets_path is not None and passage_offsets_path.exists():
        out.update(_atlas_passages_to_text(passage_offsets_path, text_by_file_id))
    else:
        logger.debug(
            "passage_offsets.json absent (%s); atlas-rag / khop_passage chunk_ids "
            "won't resolve to text.",
            passage_offsets_path,
        )

    return out


def _load_transcriptions(transcription_dir: Path) -> dict[str, str]:
    """Build ``file_id → transcription_text`` from a transcription outputs dir."""
    if not transcription_dir.exists():
        logger.warning(
            "Transcription outputs absent at %s; no passage text will resolve.",
            transcription_dir,
        )
        return {}
    out: dict[str, str] = {}
    for path in sorted(transcription_dir.glob("*.json")):
        try:
            record = EnrichedRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable transcription %s: %s", path, exc)
            continue
        out[record.file_id] = record.transcription_text
    return out


def _bm25_chunks_to_text(chunk_dir: Path, text_by_file_id: dict[str, str]) -> dict[str, str]:
    """Resolve every chunk in ``chunk_dir`` to its substring of the source transcription."""
    out: dict[str, str] = {}
    for path in sorted(chunk_dir.glob("*.json")):
        try:
            chunk_set = ChunkSet.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable ChunkSet %s: %s", path, exc)
            continue
        full_text = text_by_file_id.get(chunk_set.source_file_id)
        if full_text is None:
            logger.debug(
                "No transcription for ChunkSet %s (file_id=%s); skipping its chunks.",
                path,
                chunk_set.source_file_id,
            )
            continue
        for chunks in chunk_set.views.values():
            for chunk in chunks:
                out[chunk.chunk_id] = full_text[chunk.start_char : chunk.end_char]
    return out


def _atlas_passages_to_text(
    passage_offsets_path: Path, text_by_file_id: dict[str, str]
) -> dict[str, str]:
    """Resolve atlas-rag/khop_passage passage_ids to their source-text substrings."""
    try:
        sidecar = PassageOffsetSidecar.load(passage_offsets_path)
    except (OSError, ValueError) as exc:
        logger.warning("Skipping unreadable passage_offsets %s: %s", passage_offsets_path, exc)
        return {}
    out: dict[str, str] = {}
    for offset in sidecar.offsets:
        full_text = text_by_file_id.get(offset.source_file_id)
        if full_text is None:
            continue
        out[offset.passage_id] = full_text[offset.start_char : offset.end_char]
    return out
