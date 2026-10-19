"""OracleRetriever: gold-context upper bound for Phase C.

Returns exactly the context a QA pair was generated from: the ``cep_4k``
chunk the CEP generator read when it wrote the pair. A non-answerable item
receives the context of the CEP pair it was perturbed from, so the oracle
also measures abstention when the closest possible evidence is in context.

Retrieval is perfect by construction, so the oracle arm bounds what the
answerer can reach with the Answerer prompt held constant: its KC is the
ceiling against which the other arms are read. It mirrors the Null arm,
which bounds the scale from below.

The oracle reads ``RetrievalQuery.source_qa_pair_id`` and ignores the
question text: distinct pairs share question text, so a text lookup would be
ambiguous.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from arandu.qa.schemas import QARecordCEP, cep_qa_pair_id
from arandu.shared.rag.schemas import RetrievedPassage

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.shared.rag.schemas import RetrievalQuery

logger = logging.getLogger(__name__)


class OracleRetriever:
    """Retriever that returns the generation context of the source QA pair.

    Attributes:
        retriever_id: Stable identifier (``"oracle"``) used in `RetrievalRecord`.
    """

    retriever_id: str = "oracle"

    def __init__(self, contexts: dict[str, tuple[str, str]]) -> None:
        """Hold the ``qa_pair_id`` → ``(chunk_id, context)`` lookup.

        Args:
            contexts: For each CEP ``qa_pair_id``, the id of the chunk the
                pair was generated from and that chunk's text.
        """
        self._contexts = contexts

    @classmethod
    def from_cep_dir(cls, cep_dir: Path) -> OracleRetriever:
        """Build the lookup from a run's CEP outputs.

        Every pair is indexed, judge-rejected ones included, so ids keep the
        position-over-all-pairs scheme of :func:`cep_qa_pair_id` and a
        non-answerable item still resolves its parent.

        Args:
            cep_dir: ``results/<id>/cep/outputs/``; each ``*.json`` is a
                :class:`QARecordCEP`.

        Returns:
            An :class:`OracleRetriever` over every pair in ``cep_dir``.

        Raises:
            FileNotFoundError: If ``cep_dir`` does not exist.
        """
        if not cep_dir.exists():
            raise FileNotFoundError(
                f"CEP outputs not found at {cep_dir}. Run `arandu generate-cep-qa` first."
            )
        contexts: dict[str, tuple[str, str]] = {}
        for path in sorted(cep_dir.glob("*.json")):
            try:
                record = QARecordCEP.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValidationError) as exc:
                logger.warning("Skipping unreadable CEP file %s: %s", path, exc)
                continue
            for idx, pair in enumerate(record.qa_pairs):
                qa_pair_id = cep_qa_pair_id(record.source_file_id, pair.chunk_id, idx)
                contexts[qa_pair_id] = (pair.chunk_id or qa_pair_id, pair.context)
        return cls(contexts)

    def retrieve(self, query: RetrievalQuery, top_k: int) -> list[RetrievedPassage]:
        """Return the generation context of the query's source pair as one passage.

        The passage keeps the chunk's id (the ``cep_4k`` namespace the BM25
        arm also emits) and carries the text inline as prose, so packing and
        ``source_recovery`` need no offset resolution.

        Args:
            query: The item to retrieve for; this arm reads only
                ``query.source_qa_pair_id``.
            top_k: Maximum passages; ``0`` yields ``[]``.

        Returns:
            One passage, or ``[]`` when ``top_k`` is 0.

        Raises:
            ValueError: If ``query.source_qa_pair_id`` is ``None`` (an ad-hoc
                query has no source pair to look up).
            LookupError: If no CEP pair has that id (e.g. a
                non-answerable item generated against another CEP output).
                Raising lets the batch runner mark the item failed instead of
                writing an empty record that would read as an abstention.
        """
        if top_k < 1:
            return []
        source_qa_pair_id = query.source_qa_pair_id
        if source_qa_pair_id is None:
            raise ValueError("The oracle arm needs query.source_qa_pair_id; got None.")
        entry = self._contexts.get(source_qa_pair_id)
        if entry is None:
            raise LookupError(f"Oracle has no context for qa_pair_id {source_qa_pair_id!r}.")
        chunk_id, context = entry
        if not context.strip():
            # An empty passage would read as missing evidence, not as a failure.
            raise ValueError(f"Oracle context for qa_pair_id {source_qa_pair_id!r} is empty.")
        return [
            RetrievedPassage(
                chunk_id=chunk_id,
                rank=0,
                score=1.0,
                retriever_meta={"source_qa_pair_id": source_qa_pair_id},
                payload=context,
                payload_is_prose=True,
            )
        ]
