"""Schemas for the emic-validity pre-pass artifacts (spec §5)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class EmicScore(BaseModel):
    """One ordinal emic-validity score for a canonical-approved QA pair.

    Attributes:
        pair_index: Index of the pair within its source ``QARecordCEP``
            (a stable per-source key for the sample builder).
        bloom_level: The pair's Bloom level (carried for stratification).
        emic_score: Ordinal label in ``{1..5}``, or ``None`` if the LLM call
            errored for this pair.
        rationale: The judge's short justification.
        error: Error message when ``emic_score`` is ``None``.
    """

    pair_index: int = Field(..., ge=0)
    bloom_level: str
    emic_score: int | None
    rationale: str
    error: str | None = None


class EmicSourceScores(BaseModel):
    """Emic pre-pass scores for one source interview's approved pairs."""

    source_file_id: str
    source_filename: str
    scores: list[EmicScore]

    def save(self, path: str | Path) -> None:
        """Write the per-source scores to JSON."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EmicSourceScores:
        """Load per-source scores from JSON."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class EmicPrepassResult(BaseModel):
    """Summary of an emic pre-pass run."""

    pipeline_id: str
    sources: int
    approved_pairs: int
    scored_pairs: int
    failed_pairs: int
