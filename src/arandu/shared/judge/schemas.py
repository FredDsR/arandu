"""Judge evaluation schemas.

Domain-agnostic result types for criterion scores, step results, and
multi-stage pipeline results.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field

StageMode = Literal["filter", "score", "always"]
"""Controls how a stage affects pipeline flow.

- ``"filter"`` -- Reject stops pipeline (non-``"always"`` stages skipped).
- ``"score"``  -- Score and continue regardless.
- ``"always"`` -- Runs even after rejection by a previous stage.
"""


CriterionScale = Literal["continuous", "ordinal"]
"""Which scale a criterion's result lives on.

- ``"continuous"`` -- ``score`` holds a float in ``[0, 1]`` (default; the
  classic LLM/heuristic criterion).
- ``"ordinal"``    -- ``ordinal_score`` holds an integer label (e.g. ``{1..5}``);
  ``score`` is unused. The continuous ``threshold`` does not gate ordinal
  criteria, which run in ``score`` mode (any downstream filter threshold is
  applied separately).
"""


class CriterionScore(BaseModel):
    """Result of a single criterion evaluation.

    Carries either a continuous ``score`` or an ``ordinal_score``, selected by
    ``scale``. Continuous is the default so existing criteria are unaffected.
    """

    score: float | None = None
    ordinal_score: int | None = None
    scale: CriterionScale = "continuous"
    threshold: float
    rationale: str
    thinking: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Whether the criterion is satisfied. Always False on error.

        For continuous criteria this is ``score >= threshold``. Ordinal
        criteria run in score mode (no continuous gate), so a successful
        evaluation counts as passed; any emic filter threshold is applied
        downstream, not here.
        """
        if self.error is not None:
            return False
        if self.scale == "ordinal":
            return self.ordinal_score is not None
        if self.score is None:
            return False
        return self.score >= self.threshold


class JudgeStepResult(BaseModel):
    """Result of running all criteria in a single step."""

    criterion_scores: dict[str, CriterionScore]

    @property
    def passed(self) -> bool:
        """Whether all criteria met their thresholds."""
        return all(cs.passed for cs in self.criterion_scores.values())


class JudgePipelineResult(BaseModel):
    """Result of running the full multi-stage pipeline."""

    stage_results: dict[str, JudgeStepResult]
    passed: bool
    rejected_at: str | None = None


class JudgeResultMixin(BaseModel):
    """Mixin that adds judge verdict fields to a record schema.

    Mix into any record type that can be evaluated by a judge pipeline
    so every domain stores judge output under the same canonical name:
    the full pipeline result. ``is_valid`` is derived from
    ``validation.passed`` so the two cannot drift.
    """

    validation: JudgePipelineResult | None = Field(
        default=None,
        description=("Full judge pipeline result. None when the record has not been judged yet."),
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_valid(self) -> bool | None:
        """Pass/fail verdict derived from ``validation.passed``.

        Returns ``None`` when the record has not been judged.
        """
        return self.validation.passed if self.validation is not None else None

    @property
    def is_judge_rejected(self) -> bool:
        """Whether the record was judged and FAILED (``is_valid is False``).

        The canonical "drop this record" predicate for downstream stages that
        should consume only judge-valid records (chunk, kg). Unjudged records
        (``is_valid is None``) are NOT rejected, so a stage can still run before
        judging. Kept here as the single authoritative home for the rule that
        otherwise gets re-inlined in every consumer's batch loader.
        """
        return self.is_valid is False
