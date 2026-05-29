"""Build the ``qa_pair_id`` → gold-context join the judge needs.

The answer-judging criteria need ground-truth context that lives in the
CEP stage's output (:class:`QARecordCEP` files). This module reads those
artifacts once at batch start and produces a flat lookup keyed by the
same ``qa_pair_id`` format the retrieve / answer stages emit
(``"<file_id>:<chunk_id>:<idx>"``).

Beyond the question + gold answer, the record carries the source
``context`` the pair was generated from, which the faithfulness and
heuristic-correctness criteria score against. Analysis-only dimensions
(Bloom level, question type) are deliberately NOT carried here: they are
stratification keys, not judge inputs, and the analysis stage already
re-derives them via its own CEP cross-cut map
(:func:`arandu.shared.rag.analysis.loader.build_cross_cut_map`). The
CEP's own ``reasoning_trace`` / ``tacit_inference`` annotations are also
excluded: feeding the generator's self-explanation back as ground truth
would make the judge score conformance to the CEP annotation rather than
against an independent reference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

from arandu.qa.schemas import QARecordCEP

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class GoldRecord(BaseModel):
    """Ground-truth context for one ``qa_pair_id`` (from its CEP pair)."""

    qa_pair_id: str
    question: str
    gold_answer: str
    context: str = Field(default="", description="Source text the QA pair was generated from.")


def build_gold_lookup(cep_dir: Path) -> dict[str, GoldRecord]:
    """Build ``qa_pair_id → GoldRecord`` from CEP outputs.

    Args:
        cep_dir: ``results/<id>/cep/outputs/``. Each ``*.json`` is a
            :class:`QARecordCEP` whose ``qa_pairs`` carry questions +
            gold answers + source context + chunk_id pointers.

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
                context=pair.context,
            )
    return out
