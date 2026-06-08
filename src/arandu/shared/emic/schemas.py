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
    """Summary of an emic pre-pass run.

    Source-level counters (``completed_sources``/``resumed_sources``/
    ``failed_sources``) account for *this* invocation; ``sources`` is the total
    number of CEP source files seen. Pair-level counters
    (``approved_pairs``/``scored_pairs``/``failed_pairs``/``unjudged_pairs``)
    likewise reflect only the sources processed this run (resumed sources are
    not re-counted), so a no-op ``--resume`` legitimately reports zero pairs.

    Attributes:
        pipeline_id: Run identifier.
        sources: Total CEP source files discovered.
        completed_sources: Sources scored and persisted this invocation.
        resumed_sources: Sources skipped because already checkpointed.
        failed_sources: Sources that failed to load (skipped, no output).
        approved_pairs: Canonical-approved pairs encountered this run.
        scored_pairs: Approved pairs that received an ordinal score.
        failed_pairs: Approved pairs whose LLM call errored.
        unjudged_pairs: Pairs skipped because the run was never judged
            (``is_valid is None``); a non-zero value signals a missing
            ``arandu judge-qa`` step.
    """

    pipeline_id: str
    sources: int
    completed_sources: int = 0
    resumed_sources: int = 0
    failed_sources: int = 0
    approved_pairs: int
    scored_pairs: int
    failed_pairs: int
    unjudged_pairs: int = 0
