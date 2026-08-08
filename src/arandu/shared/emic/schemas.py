"""Schemas for the emic-validity judge artifacts (spec §5)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

EmicScope = Literal["all", "approved"]
"""Which CEP pairs the emic judge scores.

``all`` scores every pair regardless of the ``judge-qa`` verdict, so emic
validity can be cross-tabulated against approval. ``approved`` scores only
canonically-approved pairs (``is_valid`` true).
"""


class EmicScore(BaseModel):
    """One ordinal emic-validity score for a QA pair.

    Attributes:
        pair_index: Index of the pair within its source ``QARecordCEP``
            (a stable per-source key for the sample builder).
        bloom_level: The pair's Bloom level (carried for stratification).
        emic_score: Ordinal label in ``{1..5}``, or ``None`` if the LLM call
            errored for this pair.
        rationale: The judge's short justification.
        error: Error message when ``emic_score`` is ``None``.
        is_valid: The pair's ``judge-qa`` verdict at scoring time — ``True``
            approved, ``False`` rejected, ``None`` never judged. Carried so
            emic validity can be cross-tabulated against approval without
            re-joining the CEP records, and so the sample builder can restrict
            its frame to approved pairs even when the run scored everything.
    """

    pair_index: int = Field(..., ge=0)
    bloom_level: str
    emic_score: int | None
    rationale: str
    error: str | None = None
    is_valid: bool | None = None


class EmicSourceScores(BaseModel):
    """Emic-validity scores for the in-scope pairs of one source interview."""

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


class EmicJudgeResult(BaseModel):
    """Summary of an emic-validity judge run.

    Source-level counters (``completed_sources``/``resumed_sources``/
    ``failed_sources``) account for *this* invocation; ``sources`` is the total
    number of CEP source files seen. Pair-level counters likewise reflect only
    the sources processed this run (resumed sources are not re-counted), so a
    no-op ``--resume`` legitimately reports zero pairs.

    ``selected_pairs`` is the scope-dependent denominator: under ``all`` every
    pair of a processed source is selected, under ``approved`` only the
    canonically-approved ones. It always equals
    ``scored_pairs + failed_pairs``, and
    ``selected_pairs + skipped_pairs`` always equals
    ``approved_pairs + rejected_pairs + unjudged_pairs``.

    Attributes:
        pipeline_id: Run identifier.
        scope: Which pairs this run scored (see :data:`EmicScope`).
        sources: Total CEP source files discovered.
        completed_sources: Sources scored and persisted this invocation.
        resumed_sources: Sources skipped because already checkpointed.
        failed_sources: Sources that failed to load (skipped, no output).
        selected_pairs: Pairs the scope selected for scoring this run.
        scored_pairs: Selected pairs that received an ordinal score.
        failed_pairs: Selected pairs whose LLM call errored.
        skipped_pairs: Pairs excluded by the scope (always 0 under ``all``).
        approved_pairs: Pairs encountered whose ``judge-qa`` verdict was
            approve. Corpus composition, not a selection count: under
            ``approved`` it equals ``selected_pairs``, under ``all`` it is a
            subset of it.
        rejected_pairs: Pairs encountered whose ``judge-qa`` verdict was
            reject. Counted under both scopes (under ``approved`` they are
            counted and then skipped), so the corpus split stays visible
            whichever scope ran.
        unjudged_pairs: Pairs encountered with no ``judge-qa`` verdict
            (``is_valid is None``). Scored under ``all``, skipped under
            ``approved``. A non-zero value signals a missing or partial
            ``arandu judge-qa`` step.
    """

    pipeline_id: str
    scope: EmicScope
    sources: int
    completed_sources: int = 0
    resumed_sources: int = 0
    failed_sources: int = 0
    selected_pairs: int
    scored_pairs: int
    failed_pairs: int
    skipped_pairs: int = 0
    approved_pairs: int = 0
    rejected_pairs: int = 0
    unjudged_pairs: int = 0
