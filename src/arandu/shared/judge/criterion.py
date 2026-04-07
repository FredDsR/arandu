"""Base criterion protocol and implementations for LLM-as-a-Judge evaluation.

Each criterion evaluates a single aspect of generated content, returning a score
between 0.0 and 1.0 with optional rationale.
"""

from __future__ import annotations

import json
import logging
from string import Template
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from arandu.shared.judge.schemas import CriterionScore
from arandu.shared.llm_client import StructuredOutputError
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


@runtime_checkable
class JudgeCriterion(Protocol):
    """Protocol for individual evaluation criteria.

    Each criterion evaluates a single aspect of generated content using
    an LLM judge. This design follows G-Eval's approach of one criterion
    per LLM call to avoid reasoning overlap.
    """

    name: str
    threshold: float

    def evaluate(self, **kwargs: Any) -> CriterionScore:
        """Evaluate content against this criterion.

        Args:
            **kwargs: Domain-specific evaluation parameters.

        Returns:
            CriterionScore with score, rationale, and optional thinking trace.
        """
        ...


class FileCriterion:
    """File-based criterion implementation.

    Loads criterion configuration (prompt template) from a single file per language.
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

    def evaluate(self, **kwargs: Any) -> CriterionScore:
        """Evaluate content against this criterion.

        Args:
            **kwargs: Domain-specific evaluation parameters (e.g., context,
                question, answer).

        Returns:
            CriterionScore with score, rationale, and optional thinking trace.
        """
        try:
            prompt = self._build_prompt(**kwargs)

            response = self.llm_client.generate_structured(
                prompt=prompt,
                response_model=CriterionResponse,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            score = validate_score(response.score)

            return CriterionScore(
                score=score,
                threshold=self.threshold,
                rationale=response.rationale,
                thinking=None,
            )

        except StructuredOutputError as e:
            logger.warning("Criterion '%s' structured output failed: %s", self.name, e)
            return CriterionScore(
                score=None,
                threshold=self.threshold,
                rationale="",
                error=str(e),
            )

        except Exception as e:
            logger.warning("Criterion '%s' evaluation failed: %s", self.name, e)
            return CriterionScore(
                score=None,
                threshold=self.threshold,
                rationale="",
                error=str(e),
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
