"""Retriever Protocol: narrow contract for retrieval backends (spec §4.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from arandu.shared.rag.schemas import RetrievalQuery, RetrievedPassage


@runtime_checkable
class Retriever(Protocol):
    """A retrieval backend produces a ranked list of passages for an item.

    Implementations must expose a stable ``retriever_id`` and a ``retrieve``
    method that returns at most ``top_k`` :class:`RetrievedPassage` objects.
    The :class:`RetrievalQuery` carries both the question text and the item's
    provenance; each arm reads the field it keys on. The NullRetriever is a
    valid implementation: it always returns ``[]``.
    """

    retriever_id: str

    def retrieve(self, query: RetrievalQuery, top_k: int) -> list[RetrievedPassage]:
        """Retrieve up to ``top_k`` ranked passages for ``query``."""
        ...
