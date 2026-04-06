"""BaseJudge ABC for composable evaluation judges.

Provides a template-method pattern where subclasses define how to build
the pipeline and the base class handles evaluate delegation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arandu.shared.judge.pipeline import JudgePipeline
    from arandu.shared.judge.schemas import JudgePipelineResult


class BaseJudge(ABC):
    """Abstract base class for evaluation judges.

    Subclasses must implement ``_build_pipeline`` to configure the
    multi-stage evaluation pipeline.
    """

    def __init__(self) -> None:
        """Initialize judge and build its pipeline."""
        self._pipeline = self._build_pipeline()

    @abstractmethod
    def _build_pipeline(self) -> JudgePipeline:
        """Build the evaluation pipeline.

        Returns:
            Configured JudgePipeline instance.
        """
        ...

    def evaluate(self, **kwargs: Any) -> JudgePipelineResult:
        """Run evaluation through the pipeline.

        Args:
            **kwargs: Domain-specific evaluation parameters.

        Returns:
            JudgePipelineResult with per-stage results and pass/fail.
        """
        return self._pipeline.evaluate(**kwargs)
