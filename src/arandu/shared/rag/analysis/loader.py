"""Bloom + question-type lookup for analysis cross-cuts (spec §8.6).

The CEP stage's :class:`QAPairCEP` records carry ``bloom_level`` and
``question_type`` per pair. The analysis stage needs to join those
back onto judged :class:`AnswerRecord` instances by ``qa_pair_id`` to
produce Bloom-stratified and type-stratified metric tables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from arandu.qa.schemas import QARecordCEP

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CrossCutMeta(BaseModel):
    """Per-``qa_pair_id`` Bloom + question_type for stratified analysis."""

    qa_pair_id: str
    bloom_level: str
    question_type: str


def build_cross_cut_map(cep_dir: Path) -> dict[str, CrossCutMeta]:
    """Build ``qa_pair_id → (bloom_level, question_type)`` from CEP outputs.

    Args:
        cep_dir: ``results/<id>/cep/outputs/``. Each ``*.json`` is a
            :class:`QARecordCEP` whose ``qa_pairs`` carry the strata
            fields.

    Returns:
        Flat dict. Empty if ``cep_dir`` is absent (analysis falls back
        to an "all" pseudo-stratum in that case, with a warning).
    """
    out: dict[str, CrossCutMeta] = {}
    if not cep_dir.exists():
        logger.warning("CEP outputs absent at %s; cross-cut strata will be unavailable.", cep_dir)
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
            out[qa_pair_id] = CrossCutMeta(
                qa_pair_id=qa_pair_id,
                bloom_level=str(pair.bloom_level),
                question_type=str(pair.question_type),
            )
    return out
