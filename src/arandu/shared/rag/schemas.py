"""Pydantic schemas shared by the RAG retrieval, answerer, and judge stages.

Defines:

- :class:`RetrievedPassage` — a single ranked passage handed back by a retriever.
- :class:`RetrievalRecord` — the complete output of a retrieval call for one
  question against one retriever arm.
- :class:`AnswerRecord` — extends :class:`RetrievalRecord` with the Answerer's
  output and reuses Phase B's :class:`JudgeResultMixin` for verdict shape.

All three are storage-stable: serialized once at retrieval/answer/judge time
and joined in cross-arm comparison without re-running the upstream stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from arandu.shared.judge.schemas import JudgeResultMixin

if TYPE_CHECKING:
    from typing import Self


class RetrievedPassage(BaseModel):
    """A single ranked passage returned by a :class:`Retriever`.

    Attributes:
        chunk_id: Reference into the source :class:`ChunkSet` for the chunker view.
        rank: 0-indexed position in the ranked list (lower is better).
        score: Retriever-native relevance score. Sign and magnitude depend on
            the backend (BM25-Okapi, cosine, PPR-weighted), so callers must
            not compare raw scores across retriever IDs.
        retriever_meta: Backend-specific metadata (PPR weights, BM25 score
            components, embedding model, etc.). Free-form by design — never
            consumed by downstream judging.
    """

    chunk_id: str = Field(..., min_length=1)
    rank: int = Field(..., ge=0)
    score: float
    retriever_meta: dict[str, object] = Field(default_factory=dict)


class RetrievalRecord(BaseModel):
    """Complete output of one retrieval call for one question against one arm.

    Attributes:
        qa_pair_id: Composite identifier shared between :class:`QAPairCEP` and
            ``NonAnswerableItem`` (e.g. ``"<file_id>:<chunk_id>:<idx>"``).
        question: Verbatim question text as fed to the retriever.
        retriever_id: Stable identifier of the retriever arm (e.g. ``"bm25_bm25_512t"``).
        chunker_id: Identifier of the chunker view this retrieval ran against.
        top_k: Number of passages requested. ``len(passages) <= top_k``.
        passages: Ranked passages. May be empty (the NullRetriever returns ``[]``).
        elapsed_ms: Wall-clock retrieval time in milliseconds.
        is_answerable: Mirrored from the source QA item for downstream analysis.
    """

    qa_pair_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    retriever_id: str = Field(..., min_length=1)
    chunker_id: str = Field(..., min_length=1)
    top_k: int = Field(..., gt=0)
    passages: list[RetrievedPassage]
    elapsed_ms: float = Field(..., ge=0.0)
    is_answerable: bool

    @model_validator(mode="after")
    def _passages_within_top_k(self) -> Self:
        """Enforce the documented invariant ``len(passages) <= top_k``."""
        if len(self.passages) > self.top_k:
            raise ValueError(
                f"len(passages) ({len(self.passages)}) must be <= top_k ({self.top_k})"
            )
        return self

    def save(self, path: str | Path) -> None:
        """Serialize this record to ``path`` as JSON."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a record from ``path``."""
        return cls.model_validate_json(Path(path).read_text())


class AnswerRecord(RetrievalRecord, JudgeResultMixin):
    """Answerer output for one retrieval, plus optional judge verdicts.

    Inherits all :class:`RetrievalRecord` fields (so the joined record carries
    the retrieval context that produced the answer) and the
    :class:`JudgeResultMixin` for verdict storage.

    The :attr:`abstained` flag and :attr:`answer_text` are mutually constrained:
    ``answer_text is None`` if and only if ``abstained is True``.
    """

    answer_text: str | None = Field(default=None)
    abstained: bool
    rationale: str
    answerer_model: str = Field(..., min_length=1)
    answerer_temperature: float = Field(..., ge=0.0, le=2.0)
    answerer_meta: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _enforce_abstained_invariant(self) -> Self:
        """Require ``answer_text is None`` iff ``abstained is True``."""
        if self.abstained and self.answer_text is not None:
            raise ValueError("answer_text must be None when abstained is True")
        if not self.abstained and self.answer_text is None:
            raise ValueError("answer_text is required when abstained is False")
        return self
