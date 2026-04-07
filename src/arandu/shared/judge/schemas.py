"""Judge evaluation schemas.

Domain-agnostic result types for criterion scores, step results, and
multi-stage pipeline results.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

StageMode = Literal["filter", "score", "always"]
"""Controls how a stage affects pipeline flow.

- ``"filter"`` -- Reject stops pipeline (non-``"always"`` stages skipped).
- ``"score"``  -- Score and continue regardless.
- ``"always"`` -- Runs even after rejection by a previous stage.
"""


class CriterionScore(BaseModel):
    """Result of a single criterion evaluation."""

    score: float | None
    threshold: float
    rationale: str
    thinking: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Whether the score meets the threshold. Always False on error."""
        if self.error is not None or self.score is None:
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
