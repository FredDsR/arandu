"""Build the ``qa_pair_id`` → (gold answer, gold chunk) join the judge needs.

The 4 answer-judging criteria need ground-truth context that lives in
the CEP stage's output (``QARecordCEP`` files): the gold answer and
the source chunk it was generated from. This module reads those
artifacts once at batch start and produces a flat lookup keyed by
the same ``qa_pair_id`` format the retrieve / answer stages emit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from arandu.qa.schemas import QARecordCEP
from arandu.shared.chunking.schemas import Chunk, ChunkSet

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class GoldRecord(BaseModel):
    """Ground-truth context for one ``qa_pair_id``."""

    qa_pair_id: str
    question: str
    gold_answer: str
    gold_chunk: Chunk | None = None


def build_gold_lookup(cep_dir: Path, chunk_outputs_root: Path) -> dict[str, GoldRecord]:
    """Build ``qa_pair_id → GoldRecord`` from CEP outputs + ChunkSets.

    Args:
        cep_dir: ``results/<id>/cep/outputs/``. Each ``*.json`` is a
            :class:`QARecordCEP` whose ``qa_pairs`` carry questions +
            gold answers + chunk_id pointers.
        chunk_outputs_root: ``results/<id>/chunk/outputs/``. Each
            subdirectory is a chunker view (e.g. ``cep_4k/``) with
            single-view :class:`ChunkSet` files per source.

    Returns:
        Flat dict; ``gold_chunk`` is ``None`` for pairs whose
        ``QAPairCEP.chunk_id`` couldn't be resolved (rare — happens when
        the chunker view was rebuilt after CEP generation).
    """
    chunks_by_id = _build_chunks_by_id(chunk_outputs_root)
    out: dict[str, GoldRecord] = {}
    if not cep_dir.exists():
        logger.warning("CEP outputs absent at %s; gold lookup will be empty.", cep_dir)
        return out
    for path in sorted(cep_dir.glob("*.json")):
        try:
            record = QARecordCEP.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            logger.warning("Skipping unreadable CEP file %s: %s", path, exc)
            continue
        for idx, pair in enumerate(record.qa_pairs):
            chunk_id_segment = pair.chunk_id or "none"
            qa_pair_id = f"{record.source_file_id}:{chunk_id_segment}:{idx}"
            gold_chunk = chunks_by_id.get(pair.chunk_id) if pair.chunk_id else None
            out[qa_pair_id] = GoldRecord(
                qa_pair_id=qa_pair_id,
                question=pair.question,
                gold_answer=pair.answer,
                gold_chunk=gold_chunk,
            )
    return out


def _build_chunks_by_id(chunk_outputs_root: Path) -> dict[str, Chunk]:
    """Flatten every chunker view under ``chunk_outputs_root`` into ``chunk_id → Chunk``.

    Chunks are uniquely keyed by their ``chunk_id`` (sha1-based per
    PR #95's convention), so collisions across views shouldn't happen
    in practice — if they do, the last write wins and a warning fires.
    """
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
                    if chunk.chunk_id in out and out[chunk.chunk_id] != chunk:
                        logger.warning(
                            "Chunk id %s collides across views; keeping the latest.",
                            chunk.chunk_id,
                        )
                    out[chunk.chunk_id] = chunk
    return out
