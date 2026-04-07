"""BaseJudge ABC for composable evaluation judges.

Provides a template-method pattern where subclasses define how to build
the pipeline and the base class handles evaluate delegation, factory
creation, and logging.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from arandu.shared.judge.factory import JudgeCriterionFactory

if TYPE_CHECKING:
    from arandu.shared.judge.pipeline import JudgePipeline
    from arandu.shared.judge.schemas import JudgePipelineResult
    from arandu.shared.llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseJudge(ABC):
    """Abstract base class for evaluation judges.

    Subclasses must implement ``_build_pipeline`` to configure the
    multi-stage evaluation pipeline. The base class handles factory
    creation and initialization logging.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        language: str = "pt",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize judge with LLM client, build factory and pipeline.

        Args:
            llm_client: LLM client for judge evaluation.
            language: Language code for prompts (e.g., "pt", "en").
            temperature: Temperature for LLM evaluation.
            max_tokens: Maximum tokens for criterion responses.
        """
        self._factory = JudgeCriterionFactory(
            llm_client=llm_client,
            language=language,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._pipeline = self._build_pipeline()
        logger.info(
            "%s initialized with %s/%s",
            self.__class__.__name__,
            llm_client.provider.value,
            llm_client.model_id,
        )

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
