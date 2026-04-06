"""Criterion factory for managing available judge criteria.

The factory maps criterion names to their implementations and allows
different pipeline steps to request specific criteria by name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from arandu.shared.judge.criterion import FileCriterion, JudgeCriterion

if TYPE_CHECKING:
    from arandu.shared.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Default prompts directory (repo_root/prompts/judge/criteria)
DEFAULT_JUDGE_PROMPTS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "prompts" / "judge" / "criteria"
)


class JudgeCriterionFactory:
    """Factory for managing evaluation criteria.

    Allows different pipeline steps to request specific criteria by name
    (e.g., CEP uses faithfulness + bloom_calibration + informativeness +
    self_containedness; KG might use factual_accuracy + completeness).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        language: str = "pt",
        prompts_dir: Path | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize criterion factory.

        Args:
            llm_client: LLM client for criterion evaluation.
            language: Language code for prompts (e.g., "pt", "en").
            prompts_dir: Base directory for criterion prompts.
                If None, uses default.
            temperature: Temperature for LLM evaluation.
            max_tokens: Maximum tokens for criterion responses.
        """
        self.llm_client = llm_client
        self.language = language
        self.prompts_dir = prompts_dir or DEFAULT_JUDGE_PROMPTS_DIR
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._criteria: dict[str, JudgeCriterion] = {}

        logger.info(f"JudgeCriterionFactory initialized with language={language}")

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

    def register_custom_criterion(self, criterion: JudgeCriterion) -> None:
        """Register a custom criterion implementation.

        Args:
            criterion: Custom criterion instance implementing
                JudgeCriterion protocol.
        """
        self._criteria[criterion.name] = criterion
        logger.info(f"Registered custom criterion '{criterion.name}'")
