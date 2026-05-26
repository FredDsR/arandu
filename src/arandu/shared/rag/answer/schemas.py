"""Schemas for the answerer module.

:class:`AnswererOutput` is what the LLM produces (the raw JSON shape).
It's distinct from :class:`arandu.shared.rag.schemas.AnswerRecord`,
which is the persisted record that combines this output with the
upstream :class:`RetrievalRecord` provenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from typing import Self


class AnswererOutput(BaseModel):
    """Structured output from the Answerer LLM (spec §5.2).

    The Answerer is asked to return JSON with three fields. The
    consistency validator enforces the spec's mutual-exclusion rule:
    when the model abstains, ``answer`` must be ``None``; when it
    answers, ``answer`` must be non-empty.

    Attributes:
        abstained: Whether the model refused to answer (insufficient
            evidence in the passages, or the question is out of scope).
        answer: Verbatim answer text. ``None`` iff ``abstained`` is
            True; non-empty otherwise.
        rationale: Always populated. When ``abstained`` is False, this
            justifies the answer; when True, it explains what was
            missing from the passages.
    """

    abstained: bool
    answer: str | None = Field(default=None)
    rationale: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        """Enforce ``answer is None iff abstained is True``."""
        if self.abstained and self.answer is not None:
            raise ValueError("abstained=True requires answer=None")
        if not self.abstained and (self.answer is None or not self.answer.strip()):
            raise ValueError("abstained=False requires a non-empty answer")
        return self
