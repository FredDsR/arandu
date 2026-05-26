"""Build the ``qa_pair_id`` → (question, gold answer) join the judge needs.

The 4 answer-judging criteria need ground-truth context that lives in
the CEP stage's output (:class:`QARecordCEP` files): the question and
the gold answer. This module reads those artifacts once at batch start
and produces a flat lookup keyed by the same ``qa_pair_id`` format the
retrieve / answer stages emit (``"<file_id>:<chunk_id>:<idx>"``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from arandu.qa.schemas import QARecordCEP

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class GoldRecord(BaseModel):
    """Ground-truth context for one ``qa_pair_id``."""

    qa_pair_id: str
    question: str
    gold_answer: str


def build_gold_lookup(cep_dir: Path) -> dict[str, GoldRecord]:
    """Build ``qa_pair_id → GoldRecord`` from CEP outputs.

    Args:
        cep_dir: ``results/<id>/cep/outputs/``. Each ``*.json`` is a
            :class:`QARecordCEP` whose ``qa_pairs`` carry questions +
            gold answers + chunk_id pointers.

    Returns:
        Flat dict; empty if ``cep_dir`` is absent.
    """
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
            out[qa_pair_id] = GoldRecord(
                qa_pair_id=qa_pair_id,
                question=pair.question,
                gold_answer=pair.answer,
            )
    return out
