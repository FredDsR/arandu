"""Retriever Protocols — narrow contracts for retrieval backends (spec §4.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from arandu.shared.rag.schemas import RetrievedPassage


@runtime_checkable
class Retriever(Protocol):
    """A retrieval backend produces a ranked list of passages for a question.

    Implementations must expose a stable ``retriever_id`` and a ``retrieve``
    method that returns at most ``top_k`` :class:`RetrievedPassage` objects.
    The NullRetriever is a valid implementation: it always returns ``[]``.
    """

    retriever_id: str

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        """Retrieve up to ``top_k`` ranked passages for ``question``."""
        ...


@runtime_checkable
class SourceRetriever(Protocol):
    """A retrieval backend keyed on the item's provenance instead of its question.

    Used by the oracle arm, which returns the context a QA pair was generated
    from. The question text cannot key that lookup (distinct pairs share
    question text), so the batch runner passes the source ``qa_pair_id``.
    """

    retriever_id: str

    def retrieve_source(self, source_qa_pair_id: str, top_k: int) -> list[RetrievedPassage]:
        """Retrieve up to ``top_k`` passages for the pair ``source_qa_pair_id``."""
        ...
