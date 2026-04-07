"""Base criterion protocol and implementations for LLM-as-a-Judge evaluation.

Each criterion evaluates a single aspect of generated content, returning a score
between 0.0 and 1.0 with optional rationale.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from string import Template
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from arandu.shared.judge.schemas import CriterionScore
from arandu.utils.text import validate_score

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.shared.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriterionResponse(BaseModel):
    """Structured response from an LLM criterion evaluation.

    Attributes:
        score: Numeric score between 0.0 and 1.0.
        rationale: Explanation of the score.
    """

    score: float
    rationale: str


class JudgeCriterion(ABC):
    """Base class for all evaluation criteria.

    Provides error-handling wrapper around ``_evaluate_impl()``.
    Subclasses implement the actual evaluation logic via
    ``HeuristicCriterion`` or ``LLMCriterion``.

    Attributes:
        name: Criterion identifier.
        threshold: Minimum score to pass.
    """

    name: str
    threshold: float

    def evaluate(self, **kwargs: Any) -> CriterionScore:
        """Evaluate content against this criterion.

        Delegates to ``_evaluate_impl()`` and wraps errors into
        a ``CriterionScore`` with ``score=None`` and ``error`` set.

        Args:
            **kwargs: Domain-specific evaluation parameters.

        Returns:
            CriterionScore with score, rationale, and optional error.
        """
        try:
            return self._evaluate_impl(**kwargs)
        except Exception as e:
            logger.warning("Criterion '%s' evaluation failed: %s", self.name, e)
            return CriterionScore(
                score=None,
                threshold=self.threshold,
                rationale="",
                error=str(e),
            )

    @abstractmethod
    def _evaluate_impl(self, **kwargs: Any) -> CriterionScore:
        """Run the actual evaluation logic.

        Args:
            **kwargs: Domain-specific parameters.

        Returns:
            CriterionScore with score and rationale.
        """
        ...


class HeuristicCriterion(JudgeCriterion):
    """Base for heuristic criteria that don't need an LLM.

    Subclasses implement ``_check(**kwargs)`` returning a score and rationale.
    """

    def _evaluate_impl(self, **kwargs: Any) -> CriterionScore:
        """Run heuristic check and wrap result."""
        score, rationale = self._check(**kwargs)
        return CriterionScore(
            score=score,
            threshold=self.threshold,
            rationale=rationale,
        )

    @abstractmethod
    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Run the heuristic check.

        Args:
            **kwargs: Domain-specific parameters.

        Returns:
            Tuple of (score, rationale).
        """
        ...


class LLMCriterion(JudgeCriterion):
    """File-based LLM criterion implementation.

    Loads criterion configuration (prompt template) from a single file per language
    and evaluates using an LLM client.
    """

    def __init__(
        self,
        name: str,
        prompts_dir: Path,
        language: str,
        llm_client: LLMClient,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize file-based criterion.

        Args:
            name: Criterion name (e.g., "faithfulness").
            prompts_dir: Base directory for criterion prompts.
            language: Language code (e.g., "pt", "en").
            llm_client: LLM client for evaluation.
            temperature: Temperature for LLM generation.
            max_tokens: Maximum tokens for response.
        """
        self.name = name
        self.language = language
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load prompt template (rubric is already inlined)
        criterion_dir = prompts_dir / name / language
        self.prompt_template = self._load_prompt_template(criterion_dir)

        # Load threshold from config.json at criterion level (not per-language)
        config_file = prompts_dir / name / "config.json"
        self.threshold = self._load_threshold(config_file)

        logger.debug(f"Loaded criterion '{name}' for language '{language}'")

    def _load_prompt_template(self, criterion_dir: Path) -> str:
        """Load prompt template from file.

        Args:
            criterion_dir: Directory containing the prompt file.

        Returns:
            Prompt template string.

        Raises:
            FileNotFoundError: If prompt file doesn't exist.
        """
        prompt_file = criterion_dir / "prompt.md"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        return prompt_file.read_text(encoding="utf-8")

    @staticmethod
    def _load_threshold(config_file: Path) -> float:
        """Load threshold from a criterion config.json file.

        Args:
            config_file: Path to the config.json file.

        Returns:
            Threshold value as a float.

        Raises:
            FileNotFoundError: If config.json does not exist.
            KeyError: If the 'threshold' key is missing.
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Criterion config.json not found: {config_file}")

        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)

        if "threshold" not in config:
            raise KeyError(f"'threshold' key missing in {config_file}")

        return float(config["threshold"])

    def _evaluate_impl(self, **kwargs: Any) -> CriterionScore:
        """Call LLM with structured output and return scored result.

        Args:
            **kwargs: Domain-specific evaluation parameters (e.g., context,
                question, answer).

        Returns:
            CriterionScore with score and rationale from LLM.
        """
        prompt = self._build_prompt(**kwargs)

        response = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=CriterionResponse,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return CriterionScore(
            score=validate_score(response.score),
            threshold=self.threshold,
            rationale=response.rationale,
        )

    def _build_prompt(self, **kwargs: Any) -> str:
        """Build evaluation prompt from template.

        Args:
            **kwargs: Parameters for template substitution.

        Returns:
            Formatted prompt string.
        """
        template = Template(self.prompt_template)
        return template.safe_substitute(**kwargs)
