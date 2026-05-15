"""NullRetriever — parametric arm of the Phase C four-arm decomposition (spec §4.6).

Returns no passages; forces the Answerer to fall back to parametric (LLM-internal)
knowledge. The arm-vs-null delta is the *graph lift* metric in Joel's framing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arandu.shared.rag.schemas import RetrievedPassage


class NullRetriever:
    """Retriever that always returns an empty passage list.

    Attributes:
        retriever_id: Stable identifier (``"null"``) used in `RetrievalRecord`.
    """

    retriever_id: str = "null"

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        """Return an empty list regardless of inputs.

        Args:
            question: Ignored.
            top_k: Ignored.

        Returns:
            Always ``[]``.
        """
        return []
