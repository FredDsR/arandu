"""Bloom + question-type lookup for analysis cross-cuts (spec §8.6).

The CEP stage's :class:`QAPairCEP` records carry ``bloom_level`` and
``question_type`` per pair. The analysis stage needs to join those
back onto judged :class:`AnswerRecord` instances by ``qa_pair_id`` to
produce Bloom-stratified and type-stratified metric tables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from arandu.qa.schemas import iter_cep_pairs

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
        Flat dict. Empty if ``cep_dir`` is absent or no files parse;
        the batch layer then emits the joint table only (no Bloom or
        question-type sub-tables) and logs a warning.
    """
    out: dict[str, CrossCutMeta] = {}
    if not cep_dir.exists():
        logger.warning("CEP outputs absent at %s; cross-cut strata will be unavailable.", cep_dir)
        return out
    for ref in iter_cep_pairs(cep_dir):
        out[ref.qa_pair_id] = CrossCutMeta(
            qa_pair_id=ref.qa_pair_id,
            bloom_level=str(ref.pair.bloom_level),
            question_type=str(ref.pair.question_type),
        )
    return out
