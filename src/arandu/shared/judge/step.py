"""JudgeStep -- runs N criteria with individual thresholds.

Evaluates content against multiple criteria independently (G-Eval style,
one evaluation per criterion). Each criterion score is compared against
its configured threshold to determine pass/fail.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from arandu.shared.judge.schemas import CriterionScore, JudgeStepResult

if TYPE_CHECKING:
    from arandu.shared.judge.criterion import JudgeCriterion

logger = logging.getLogger(__name__)


class JudgeStep:
    """Runs multiple criteria and checks individual thresholds.

    Args:
        criteria: List of criteria to evaluate.
        thresholds: Minimum score per criterion name. Missing entries default to 0.0.
    """

    def __init__(
        self,
        criteria: list[JudgeCriterion],
        thresholds: dict[str, float],
    ) -> None:
        self._criteria = criteria
        self._thresholds = thresholds

    def evaluate(self, **kwargs: Any) -> JudgeStepResult:
        """Evaluate all criteria and return results with thresholds.

        Args:
            **kwargs: Domain-specific fields forwarded to each criterion.

        Returns:
            JudgeStepResult with per-criterion scores and pass/fail.
        """
        scores: dict[str, CriterionScore] = {}

        for criterion in self._criteria:
            score = criterion.evaluate(**kwargs)
            threshold = self._thresholds.get(criterion.name, 0.0)
            score.threshold = threshold
            scores[criterion.name] = score

        return JudgeStepResult(criterion_scores=scores)
