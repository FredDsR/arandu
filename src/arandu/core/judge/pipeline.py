"""Judge pipeline for multi-criterion evaluation.

Orchestrates evaluation across multiple criteria with configurable weights
and aggregates results into an overall score.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from arandu.qa.schemas import CriterionScore, ValidationScore

if TYPE_CHECKING:
    from arandu.core.judge.criterion import JudgeCriterion

logger = logging.getLogger(__name__)


class JudgePipeline:
    """Pipeline for executing multi-criterion evaluation.

    Evaluates content against multiple criteria independently, then combines
    scores using configurable weights. This approach follows G-Eval's design
    to avoid reasoning overlap between criteria.
    """

    def __init__(
        self,
        criteria: list[JudgeCriterion],
        weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize judge pipeline.

        Args:
            criteria: List of criteria to evaluate.
            weights: Optional weights for criteria (must sum to 1.0).
                    If None, uses equal weights for all criteria.

        Raises:
            ValueError: If weights don't sum to 1.0 or missing criteria.
        """
        self.criteria = criteria

        # Set default equal weights if not provided
        if weights is None:
            weight_value = 1.0 / len(criteria) if criteria else 1.0
            self.weights = {c.name: weight_value for c in criteria}
        else:
            self.weights = weights

        # Validate weights
        self._validate_weights()

        logger.info(
            f"JudgePipeline initialized with {len(criteria)} criteria: {[c.name for c in criteria]}"
        )

    def _validate_weights(self) -> None:
        """Validate that weights are properly configured.

        Raises:
            ValueError: If weights don't sum to 1.0 or missing criteria.
        """
        criterion_names = {c.name for c in self.criteria}
        weight_names = set(self.weights.keys())

        # Check all criteria have weights
        if criterion_names != weight_names:
            missing = criterion_names - weight_names
            extra = weight_names - criterion_names
            msg_parts = []
            if missing:
                msg_parts.append(f"missing weights: {sorted(missing)}")
            if extra:
                msg_parts.append(f"extra weights: {sorted(extra)}")
            raise ValueError(f"Weight mismatch - {', '.join(msg_parts)}")

        # Check weights sum to 1.0 (allow small floating point errors)
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}: {self.weights}")

    def evaluate(
        self,
        context: str,
        question: str,
        answer: str,
        **extra_params: Any,
    ) -> ValidationScore:
        """Evaluate content against all criteria.

        Each criterion is evaluated independently in a separate LLM call
        to avoid reasoning overlap (G-Eval approach).

        Args:
            context: Source context for grounding check.
            question: Question being evaluated.
            answer: Answer being evaluated.
            **extra_params: Dict of criterion-specific extra parameters,
                           keyed by criterion name.

        Returns:
            ValidationScore with individual scores and overall weighted score.
        """
        criterion_scores: dict[str, CriterionScore] = {}

        # Evaluate each criterion independently
        for criterion in self.criteria:
            # Get criterion-specific params if provided
            criterion_params = extra_params.get(criterion.name, {})

            # Evaluate
            score = criterion.evaluate(
                context=context,
                question=question,
                answer=answer,
                **criterion_params,
            )
            criterion_scores[criterion.name] = score

            logger.debug(
                f"Criterion '{criterion.name}': score={score.score:.2f}, "
                f"rationale={score.rationale[:50] if score.rationale else 'None'}..."
            )

        # Calculate weighted overall score
        overall_score = self._calculate_overall_score(criterion_scores)

        # Build ValidationScore with backward compatibility
        # (map criterion scores to existing ValidationScore fields)
        return self._build_validation_score(criterion_scores, overall_score)

    def _calculate_overall_score(
        self,
        criterion_scores: dict[str, CriterionScore],
    ) -> float:
        """Calculate weighted overall score.

        Args:
            criterion_scores: Dict of criterion scores.

        Returns:
            Weighted overall score in [0.0, 1.0].
        """
        overall = sum(
            criterion_scores[name].score * weight for name, weight in self.weights.items()
        )
        return max(0.0, min(1.0, overall))  # Clamp to valid range

    def _build_validation_score(
        self,
        criterion_scores: dict[str, CriterionScore],
        overall_score: float,
    ) -> ValidationScore:
        """Build ValidationScore from criterion results.

        Maps criterion scores to ValidationScore fields for backward compatibility
        with existing CEP validation.

        Args:
            criterion_scores: Dict of criterion scores.
            overall_score: Weighted overall score.

        Returns:
            ValidationScore with all fields populated.
        """
        # Extract individual criterion scores with defaults
        faithfulness = criterion_scores.get("faithfulness")
        bloom_calibration = criterion_scores.get("bloom_calibration")
        informativeness = criterion_scores.get("informativeness")
        self_containedness = criterion_scores.get("self_containedness")

        # Combine rationales and thinking traces
        rationales = [
            f"{name}: {score.rationale}"
            for name, score in criterion_scores.items()
            if score.rationale
        ]
        judge_rationale = "\n".join(rationales) if rationales else None

        # Collect thinking traces
        thinking_traces = [
            f"[{name}]\n{score.thinking}"
            for name, score in criterion_scores.items()
            if score.thinking
        ]
        judge_thinking = "\n\n".join(thinking_traces) if thinking_traces else None

        return ValidationScore(
            faithfulness=faithfulness.score if faithfulness else 0.5,
            bloom_calibration=bloom_calibration.score if bloom_calibration else 0.5,
            informativeness=informativeness.score if informativeness else 0.5,
            self_containedness=self_containedness.score if self_containedness else 1.0,
            overall_score=overall_score,
            judge_rationale=judge_rationale,
            judge_thinking=judge_thinking,
            criterion_scores=criterion_scores,
        )
