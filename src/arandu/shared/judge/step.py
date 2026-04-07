"""JudgeStep -- runs N criteria with individual thresholds.

Evaluates content against multiple criteria independently (G-Eval style,
one evaluation per criterion). Each criterion carries its own threshold
from its config.json.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from arandu.shared.judge.schemas import CriterionScore, JudgeStepResult

if TYPE_CHECKING:
    from arandu.shared.judge.criterion import JudgeCriterion
    from arandu.shared.judge.factory import JudgeCriterionFactory

logger = logging.getLogger(__name__)


class JudgeStep:
    """Runs multiple criteria and checks individual thresholds.

    Criteria can be provided as ``JudgeCriterion`` objects or as plain
    strings.  Strings are resolved via *factory*.get_criterion(name).
    """

    def __init__(
        self,
        criteria: list[JudgeCriterion | str],
        factory: JudgeCriterionFactory | None = None,
    ) -> None:
        """Initialize the step with criteria (objects or names).

        Args:
            criteria: Criterion objects or string names to resolve.
            factory: Factory required when *criteria* contains strings.

        Raises:
            ValueError: If a string criterion is given without a factory.
        """
        self._criteria = self._resolve_criteria(criteria, factory)

    @staticmethod
    def _resolve_criteria(
        criteria: list[JudgeCriterion | str],
        factory: JudgeCriterionFactory | None,
    ) -> list[JudgeCriterion]:
        """Convert string criterion names to objects via factory.

        Args:
            criteria: Mixed list of objects and/or string names.
            factory: Factory for resolving strings.

        Returns:
            List of resolved JudgeCriterion instances.

        Raises:
            ValueError: If a string is present but no factory was given.
        """
        resolved: list[JudgeCriterion] = []
        for item in criteria:
            if isinstance(item, str):
                if factory is None:
                    raise ValueError(
                        f"String criterion {item!r} requires a factory, "
                        "but no factory was provided."
                    )
                resolved.append(factory.get_criterion(item))
            else:
                resolved.append(item)
        return resolved

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
            scores[criterion.name] = score.model_copy(update={"threshold": criterion.threshold})

        return JudgeStepResult(criterion_scores=scores)
