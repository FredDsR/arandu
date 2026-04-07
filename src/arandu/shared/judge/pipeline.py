"""JudgePipeline -- multi-stage evaluation with filter/score/always modes.

Orchestrates a sequence of ``JudgeStage`` instances. Each stage wraps a
``JudgeStep`` and declares a ``StageMode`` that controls how its result
affects the pipeline:

- **filter** -- a failing step causes rejection; subsequent non-always
  stages are skipped.
- **score** -- the result is recorded but never causes rejection.
- **always** -- the step runs even after a prior rejection.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from arandu.shared.judge.schemas import JudgePipelineResult, JudgeStepResult, StageMode
from arandu.shared.judge.step import JudgeStep  # noqa: TC001 (Pydantic needs runtime access)

logger = logging.getLogger(__name__)


class JudgeStage(BaseModel):
    """A named pipeline stage that pairs a step with a run mode.

    Attributes:
        name: Human-readable identifier for the stage.
        step: The evaluation step to execute.
        mode: Controls how the stage affects pipeline flow.
    """

    name: str
    step: JudgeStep
    mode: StageMode = "score"

    model_config = {"arbitrary_types_allowed": True}


class JudgePipeline:
    """Multi-stage evaluation pipeline.

    Runs stages sequentially. Filter stages that fail cause rejection
    and skip subsequent non-always stages.

    Args:
        stages: Ordered list of stages to evaluate.
    """

    def __init__(self, stages: list[JudgeStage]) -> None:
        self._stages = stages

    def evaluate(self, **kwargs: Any) -> JudgePipelineResult:
        """Run all stages and return aggregated results.

        Args:
            **kwargs: Domain-specific fields forwarded to each stage's step.

        Returns:
            JudgePipelineResult with per-stage results, pass/fail, and
            optional rejection point.
        """
        stage_results: dict[str, JudgeStepResult] = {}
        rejected = False
        rejected_at: str | None = None

        for stage in self._stages:
            if rejected and stage.mode != "always":
                logger.debug(
                    "Skipping stage '%s' (pipeline rejected at '%s')",
                    stage.name,
                    rejected_at,
                )
                continue

            step_result = stage.step.evaluate(**kwargs)
            stage_results[stage.name] = step_result

            if stage.mode == "filter" and not step_result.passed:
                rejected = True
                rejected_at = stage.name
                logger.info("Pipeline rejected at filter stage '%s'", stage.name)

        return JudgePipelineResult(
            stage_results=stage_results,
            passed=not rejected,
            rejected_at=rejected_at,
        )
