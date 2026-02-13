"""Base criterion protocol and implementations for LLM-as-a-Judge evaluation.

Each criterion evaluates a single aspect of generated content, returning a score
between 0.0 and 1.0 with optional rationale.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, Protocol

from gtranscriber.schemas import CriterionScore

if TYPE_CHECKING:
    from gtranscriber.core.llm_client import LLMClient

logger = logging.getLogger(__name__)


class JudgeCriterion(Protocol):
    """Protocol for individual evaluation criteria.

    Each criterion evaluates a single aspect of generated content using
    an LLM judge. This design follows G-Eval's approach of one criterion
    per LLM call to avoid reasoning overlap.
    """

    name: str
    rubric: str
    prompt_template: str

    def evaluate(
        self,
        context: str,
        question: str,
        answer: str,
        **extra_params: Any,
    ) -> CriterionScore:
        """Evaluate content against this criterion.

        Args:
            context: Source context for grounding check.
            question: Question being evaluated.
            answer: Answer being evaluated.
            **extra_params: Additional criterion-specific parameters.

        Returns:
            CriterionScore with score, rationale, and optional thinking trace.
        """
        ...


class FileCriterion:
    """File-based criterion implementation.

    Loads criterion configuration (rubric, prompt template) from files.
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

        # Load criterion configuration
        criterion_dir = prompts_dir / name / language
        self.rubric, self.prompt_template = self._load_criterion_files(criterion_dir)

        logger.debug(f"Loaded criterion '{name}' for language '{language}'")

    def _load_criterion_files(self, criterion_dir: Path) -> tuple[str, str]:
        """Load rubric and prompt template from files.

        Args:
            criterion_dir: Directory containing criterion files.

        Returns:
            Tuple of (rubric, prompt_template).

        Raises:
            FileNotFoundError: If criterion files don't exist.
        """
        rubric_file = criterion_dir / "rubric.md"
        prompt_file = criterion_dir / "prompt.md"

        if not rubric_file.exists():
            raise FileNotFoundError(f"Rubric file not found: {rubric_file}")
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        rubric = rubric_file.read_text(encoding="utf-8")
        prompt_template = prompt_file.read_text(encoding="utf-8")

        return rubric, prompt_template

    def evaluate(
        self,
        context: str,
        question: str,
        answer: str,
        **extra_params: Any,
    ) -> CriterionScore:
        """Evaluate content against this criterion.

        Args:
            context: Source context for grounding check.
            question: Question being evaluated.
            answer: Answer being evaluated.
            **extra_params: Additional criterion-specific parameters.

        Returns:
            CriterionScore with score, rationale, and optional thinking trace.
        """
        try:
            # Build evaluation prompt
            prompt = self._build_prompt(context, question, answer, **extra_params)

            # Call LLM judge
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Parse response
            score = self._parse_response(result.content, thinking=result.thinking)
            return score

        except Exception as e:
            logger.warning(f"Criterion '{self.name}' evaluation failed: {e}")
            # Return default neutral score on error
            return CriterionScore(
                criterion_name=self.name,
                score=0.5,
                rationale=f"Evaluation failed: {e}",
                thinking=None,
            )

    def _build_prompt(
        self,
        context: str,
        question: str,
        answer: str,
        **extra_params: Any,
    ) -> str:
        """Build evaluation prompt from template.

        Args:
            context: Source context.
            question: Question being evaluated.
            answer: Answer being evaluated.
            **extra_params: Additional parameters for template substitution.

        Returns:
            Formatted prompt string.
        """
        template = Template(self.prompt_template)
        # Merge standard params with extra params
        params = {
            "context": context,
            "question": question,
            "answer": answer,
            "rubric": self.rubric,
            **extra_params,
        }
        return template.safe_substitute(params)

    def _parse_response(
        self,
        response: str,
        *,
        thinking: str | None = None,
    ) -> CriterionScore:
        """Parse LLM response into CriterionScore.

        Expected JSON format:
        {
            "score": 0.8,
            "rationale": "Explanation of the score"
        }

        Args:
            response: Raw LLM response.
            thinking: Optional thinking trace from judge model.

        Returns:
            CriterionScore with parsed values.
        """
        response = response.strip()
        # Strip markdown code blocks if present
        if response.startswith("```"):
            response = re.sub(r"```(?:json)?\n?", "", response)

        try:
            data = json.loads(response)
            score = self._validate_score(data.get("score", 0.5))
            rationale = data.get("rationale") or data.get("explanation")

            return CriterionScore(
                criterion_name=self.name,
                score=score,
                rationale=rationale,
                thinking=thinking,
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {self.name} response: {e}")
            return CriterionScore(
                criterion_name=self.name,
                score=0.5,
                rationale=f"Failed to parse judge response: {e}",
                thinking=thinking,
            )

    def _validate_score(self, value: Any) -> float:
        """Validate and clamp score to [0.0, 1.0].

        Args:
            value: Raw score value.

        Returns:
            Validated float score in [0.0, 1.0].
        """
        try:
            score = float(value)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.5
