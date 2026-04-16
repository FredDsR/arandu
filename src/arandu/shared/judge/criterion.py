"""Base criterion classes and implementations for LLM-as-a-Judge evaluation.

Provides a three-level hierarchy:

- ``JudgeCriterion`` — ABC with shared error handling in ``evaluate()``.
- ``HeuristicCriterion`` — for pure-Python checks (no LLM).
- ``LLMCriterion`` — for LLM-based evaluation with prompt templates.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from string import Template
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from pydantic import BaseModel, Field

from arandu.shared.judge.schemas import CriterionScore
from arandu.utils.text import validate_score

logger = logging.getLogger(__name__)


class CriterionConfig(BaseModel):
    """Base configuration for any criterion."""

    threshold: float = Field(ge=0.0, le=1.0)


class LLMCriterionConfig(CriterionConfig):
    """Configuration for LLM-based criteria.

    Loaded from ``config.json`` at criterion level. Fields override
    factory defaults when present.
    """

    temperature: float | None = None

    @classmethod
    def load(cls, config_file: Path) -> LLMCriterionConfig:
        """Load config from a JSON file.

        Args:
            config_file: Path to config.json.

        Returns:
            Validated config instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Criterion config not found: {config_file}")
        return cls.model_validate_json(config_file.read_text(encoding="utf-8"))


class CriterionResponse(BaseModel):
    """Expected structured response from an LLM criterion evaluation."""

    score: float
    rationale: str


class JudgeCriterion(ABC):
    """Base class for all evaluation criteria.

    Provides error-handling wrapper around ``_evaluate_impl()``.
    Subclasses implement the actual evaluation logic via
    ``HeuristicCriterion`` or ``LLMCriterion``.

    Args:
        name: Criterion identifier.
        threshold: Minimum score to pass.
    """

    def __init__(self, name: str, threshold: float) -> None:
        self.name = name
        self.threshold = threshold

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

    Args:
        name: Criterion identifier.
        threshold: Minimum score to pass.
    """

    def _evaluate_impl(self, **kwargs: Any) -> CriterionScore:
        """Run heuristic check and wrap result."""
        score, rationale = self._check(**kwargs)
        return CriterionScore(
            score=validate_score(score),
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
    """LLM-based criterion using prompt templates and structured output.

    Use ``from_config()`` to load from prompt files and config.json,
    or instantiate directly for testing.

    Args:
        name: Criterion identifier.
        threshold: Minimum score to pass.
        llm_client: LLM client for evaluation.
        prompt_template: Prompt template string with ``$variable`` placeholders.
        temperature: Sampling temperature for LLM generation.
        max_tokens: Maximum tokens for response.
    """

    def __init__(
        self,
        name: str,
        threshold: float,
        llm_client: Any,
        prompt_template: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        super().__init__(name, threshold)
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens

    @classmethod
    def from_config(
        cls,
        name: str,
        prompts_dir: Path,
        language: str,
        llm_client: Any,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> LLMCriterion:
        """Load criterion from prompt files and config.json.

        Args:
            name: Criterion name (e.g., "faithfulness").
            prompts_dir: Base directory for criterion prompts.
            language: Language code (e.g., "pt", "en").
            llm_client: LLM client for evaluation.
            temperature: Default temperature (overridden by config if set).
            max_tokens: Maximum tokens for response.

        Returns:
            Configured LLMCriterion instance.
        """
        config = LLMCriterionConfig.load(prompts_dir / name / "config.json")

        prompt_file = prompts_dir / name / language / "prompt.md"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        prompt_template = prompt_file.read_text(encoding="utf-8")

        # Config temperature overrides factory default if set
        effective_temp = config.temperature if config.temperature is not None else temperature

        logger.debug("Loaded criterion '%s' for language '%s'", name, language)

        return cls(
            name=name,
            threshold=config.threshold,
            llm_client=llm_client,
            prompt_template=prompt_template,
            temperature=effective_temp,
            max_tokens=max_tokens,
        )

    def _evaluate_impl(self, **kwargs: Any) -> CriterionScore:
        """Call LLM with structured output and return scored result.

        Args:
            **kwargs: Domain-specific evaluation parameters.

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
