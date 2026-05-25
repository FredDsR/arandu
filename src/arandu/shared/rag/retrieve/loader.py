"""Load CEP QA + (when present) non-answerable items as a uniform question stream.

The retrieve stage iterates two source datasets:

- **CEP QA**: per-source ``QARecordCEP`` files at ``results/<id>/cep/outputs/*.json``.
  Each record holds many :class:`QAPairCEP` items; we iterate them.
- **Non-answerable** (optional, when the ``nonanswerable`` stage has
  populated): at ``results/<id>/nonanswerable/outputs/`` — silently
  skipped when absent (that stage is still in development).

Both produce :class:`QuestionRecord` rows, distinguished by ``source``
(``"cep"`` or ``"nonanswerable"``). The retrieve batch runner emits
outputs into separate subdirectories per the spec §10.4 amendment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from arandu.qa.schemas import QARecordCEP

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)


QuestionSource = Literal["cep", "nonanswerable"]


class QuestionRecord(BaseModel):
    """One question to retrieve against, normalized across source types.

    Attributes:
        qa_pair_id: Composite identifier per :class:`RetrievalRecord`'s
            contract (``"<file_id>:<chunk_id>:<idx>"``). Unique within a run.
        question: Verbatim question text to feed the retriever.
        is_answerable: Mirrored from the source item. CEP pairs are
            answerable by construction; non-answerable items are not.
        source: ``"cep"`` or ``"nonanswerable"`` — drives the output
            subdirectory under ``retrieve/outputs/<arm_id>/``.
        source_file_id: The source media file id the item is associated with.
            Useful for downstream grouping but not used by the retriever
            itself.
        chunker_id: Chunker view recorded on the source item (CEP records
            track which view was used to slice the transcription for QA
            generation). The retrieve stage records this on each
            :class:`RetrievalRecord` to preserve the chunker → retriever
            provenance.
    """

    qa_pair_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    is_answerable: bool
    source: QuestionSource
    source_file_id: str
    chunker_id: str


def load_questions(cep_dir: Path, nonanswerable_dir: Path | None = None) -> list[QuestionRecord]:
    """Walk the CEP + non-answerable output directories into a flat question list.

    Args:
        cep_dir: ``results/<id>/cep/outputs/``. Each ``*.json`` is a
            :class:`QARecordCEP`. ``FileNotFoundError`` if missing —
            CEP is required for any retrieve run.
        nonanswerable_dir: ``results/<id>/nonanswerable/outputs/``. When
            ``None`` or non-existent, silently skipped (the stage that
            populates this is still in development).

    Returns:
        Flat list of :class:`QuestionRecord` rows. CEP rows first, in
        sorted file order; then non-answerable. Stable ordering so
        checkpoint resume + cross-run diffing are deterministic.

    Raises:
        FileNotFoundError: If ``cep_dir`` does not exist.
    """
    if not cep_dir.exists():
        raise FileNotFoundError(
            f"CEP outputs not found at {cep_dir}. Run `arandu generate-cep-qa` first."
        )

    questions: list[QuestionRecord] = list(_iter_cep_questions(cep_dir))
    if nonanswerable_dir is not None and nonanswerable_dir.exists():
        questions.extend(_iter_nonanswerable_questions(nonanswerable_dir))
    else:
        logger.debug(
            "Non-answerable dir absent (%s); only CEP questions will be retrieved.",
            nonanswerable_dir,
        )
    return questions


def _iter_cep_questions(cep_dir: Path) -> Iterator[QuestionRecord]:
    """Yield one :class:`QuestionRecord` per ``QAPairCEP`` in the dir.

    Each on-disk file is a :class:`QARecordCEP` holding many pairs. The
    composite ``qa_pair_id`` follows the schema's documented form:
    ``"<file_id>:<chunk_id or 'none'>:<idx>"``. ``idx`` is the pair's
    position within the record, so the id stays stable across reruns
    that produce the same file.
    """
    for path in sorted(cep_dir.glob("*.json")):
        try:
            record = QARecordCEP.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping unreadable CEP file %s: %s", path, exc)
            continue
        for idx, pair in enumerate(record.qa_pairs):
            chunk_id_segment = pair.chunk_id or "none"
            yield QuestionRecord(
                qa_pair_id=f"{record.source_file_id}:{chunk_id_segment}:{idx}",
                question=pair.question,
                is_answerable=True,
                source="cep",
                source_file_id=record.source_file_id,
                chunker_id=record.chunker_id,
            )


def _iter_nonanswerable_questions(nonanswerable_dir: Path) -> Iterator[QuestionRecord]:
    """Yield non-answerable items as :class:`QuestionRecord`.

    Defensive shape: the non-answerable stage doesn't exist on disk yet,
    so the schema we load against is the spec-described one but read
    permissively (skip unrecognized files). Once
    ``arandu generate-non-answerable`` lands and pins the schema, this
    helper can tighten.
    """
    import json

    for path in sorted(nonanswerable_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable non-answerable file %s: %s", path, exc)
            continue
        for item in payload.get("items", []):
            qa_pair_id = item.get("qa_pair_id") or item.get("id")
            question = item.get("question")
            source_file_id = item.get("source_file_id") or item.get("source_gdrive_id")
            chunker_id = item.get("chunker_id", "unknown")
            if not (qa_pair_id and question and source_file_id):
                logger.debug("Skipping incomplete non-answerable item in %s", path)
                continue
            yield QuestionRecord(
                qa_pair_id=qa_pair_id,
                question=question,
                is_answerable=False,
                source="nonanswerable",
                source_file_id=source_file_id,
                chunker_id=chunker_id,
            )
