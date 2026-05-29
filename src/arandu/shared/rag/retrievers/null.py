"""NullRetriever: abstention/hallucination baseline for Phase C (spec §4.6).

Returns no passages. With the Answerer prompt held constant across all retrieval
arms, the Null arm is expected to abstain on most items; cases where it commits
despite zero passages surface as parametric-memory leaks. Used as the
abstention/hallucination baseline for cross-arm comparison via abstention-F1 and
hallucination-rate (not via a KC subtraction; the corpus is private ethnographic
material outside LLM parametric coverage by construction).
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
