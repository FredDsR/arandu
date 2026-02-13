"""Criterion registry for managing available judge criteria.

The registry maps criterion names to their implementations and allows
different pipeline steps to request specific criteria combinations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from gtranscriber.core.judge.criterion import FileCriterion, JudgeCriterion

if TYPE_CHECKING:
    from gtranscriber.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Default prompts directory (repo_root/prompts/judge/criteria)
DEFAULT_JUDGE_PROMPTS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "prompts" / "judge" / "criteria"
)


class JudgeRegistry:
    """Registry for managing evaluation criteria.

    Allows different pipeline steps to request specific criteria combinations
    (e.g., CEP uses faithfulness + bloom_calibration + informativeness +
    self_containedness; KG might use factual_accuracy + completeness).
    """

    # Predefined criterion sets for different pipeline steps
    CRITERION_SETS = {
        "cep_validation": [
            "faithfulness",
            "bloom_calibration",
            "informativeness",
            "self_containedness",
        ],
        # Future: Add other pipeline criterion sets
        # "kg_validation": ["factual_accuracy", "completeness"],
    }

    def __init__(
        self,
        llm_client: LLMClient,
        language: str = "pt",
        prompts_dir: Path | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize criterion registry.

        Args:
            llm_client: LLM client for criterion evaluation.
            language: Language code for prompts (e.g., "pt", "en").
            prompts_dir: Base directory for criterion prompts. If None, uses default.
            temperature: Temperature for LLM evaluation.
            max_tokens: Maximum tokens for criterion responses.
        """
        self.llm_client = llm_client
        self.language = language
        self.prompts_dir = prompts_dir or DEFAULT_JUDGE_PROMPTS_DIR
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._criteria: dict[str, JudgeCriterion] = {}

        logger.info(f"JudgeRegistry initialized with language={language}")

    def get_criterion(self, name: str) -> JudgeCriterion:
        """Get or create a criterion by name.

        Args:
            name: Criterion name (e.g., "faithfulness").

        Returns:
            JudgeCriterion instance.

        Raises:
            FileNotFoundError: If criterion files don't exist.
        """
        # Return cached criterion if available
        if name in self._criteria:
            return self._criteria[name]

        # Create new file-based criterion
        criterion = FileCriterion(
            name=name,
            prompts_dir=self.prompts_dir,
            language=self.language,
            llm_client=self.llm_client,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Cache for reuse
        self._criteria[name] = criterion
        logger.debug(f"Registered criterion '{name}'")

        return criterion

    def get_criteria(self, criterion_set: str) -> list[JudgeCriterion]:
        """Get multiple criteria for a pipeline step.

        Args:
            criterion_set: Name of predefined criterion set (e.g., "cep_validation").

        Returns:
            List of JudgeCriterion instances.

        Raises:
            ValueError: If criterion set is not defined.
            FileNotFoundError: If any criterion files don't exist.
        """
        if criterion_set not in self.CRITERION_SETS:
            raise ValueError(
                f"Unknown criterion set: {criterion_set!r}. "
                f"Available sets: {sorted(self.CRITERION_SETS.keys())}"
            )

        criterion_names = self.CRITERION_SETS[criterion_set]
        return [self.get_criterion(name) for name in criterion_names]

    def register_custom_criterion(self, criterion: JudgeCriterion) -> None:
        """Register a custom criterion implementation.

        Args:
            criterion: Custom criterion instance implementing JudgeCriterion protocol.
        """
        self._criteria[criterion.name] = criterion
        logger.info(f"Registered custom criterion '{criterion.name}'")
