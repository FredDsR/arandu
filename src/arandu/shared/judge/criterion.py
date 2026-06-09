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
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

from pydantic import BaseModel, Field

from arandu.shared.judge.schemas import CriterionScale, CriterionScore
from arandu.shared.llm_settings import REASONING_MODEL_MAX_TOKENS
from arandu.utils.text import validate_ordinal_score, validate_score

ORDINAL_MIN = 1
ORDINAL_MAX = 5

# Default completion-token budget for LLM criteria. Aliases the shared
# REASONING_MODEL_MAX_TOKENS so the judge tracks the same headroom as every
# other stage (reasoning models' thinking tokens count against this budget; a
# tight cap truncates the JSON verdict mid-string into a JSONDecodeError).
DEFAULT_MAX_TOKENS = REASONING_MODEL_MAX_TOKENS

logger = logging.getLogger(__name__)


def _render_prompt(prompt_template: str, **kwargs: Any) -> str:
    """Render a ``$variable`` prompt template with the given fields.

    Shared by the template-based LLM criteria (continuous and ordinal).

    Args:
        prompt_template: Template string with ``$variable`` placeholders.
        **kwargs: Values substituted into the template.

    Returns:
        Formatted prompt string.
    """
    return Template(prompt_template).safe_substitute(**kwargs)


class BaseCriterionConfig(BaseModel):
    """Shared configuration for criteria loaded from ``config.json``.

    Carries the optional sampling temperature and the JSON loader. Concrete
    configs add their own fields (e.g. a pass ``threshold`` for filtering
    criteria); ordinal criteria need nothing further.
    """

    temperature: float | None = None

    @classmethod
    def load(cls, config_file: Path) -> Self:
        """Load config from a JSON file.

        Args:
            config_file: Path to config.json.

        Returns:
            Validated config instance of the calling subclass.

        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Criterion config not found: {config_file}")
        return cls.model_validate_json(config_file.read_text(encoding="utf-8"))


class CriterionConfig(BaseCriterionConfig):
    """Base configuration for filtering criteria (carries a pass threshold)."""

    threshold: float = Field(ge=0.0, le=1.0)


class LLMCriterionConfig(CriterionConfig):
    """Configuration for LLM-based filtering criteria.

    Loaded from ``config.json`` at criterion level. Fields override
    factory defaults when present.
    """


class RangeCriterionResponse(BaseModel):
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

    @property
    def scale(self) -> CriterionScale:
        """Scale of this criterion's score. Continuous unless overridden.

        Used by the pipeline to reject score-mode-only (ordinal) criteria from
        filter stages.
        """
        return "continuous"

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


class OrdinalCriterionResponse(BaseModel):
    """Expected structured response from an ordinal LLM criterion.

    ``score`` is an integer label on the ordinal scale ``[ORDINAL_MIN,
    ORDINAL_MAX]``. The range constraint makes ``generate_structured`` retry
    when the model returns an out-of-range or fractional value.
    """

    score: int = Field(ge=ORDINAL_MIN, le=ORDINAL_MAX)
    rationale: str


class OrdinalCriterionConfig(BaseCriterionConfig):
    """Configuration for ordinal LLM criteria.

    Ordinal criteria run in score mode, so no continuous ``threshold`` is
    required; only the optional sampling temperature (from the shared base)
    is configurable.
    """


class BaseLLMCriterion(JudgeCriterion):
    """Shared machinery for prompt-template LLM criteria.

    Renders a ``$variable`` prompt, calls the structured LLM endpoint with
    ``RESPONSE_MODEL``, and maps the response to a ``CriterionScore`` via
    ``_score_from_response``. Concrete engines set the class attributes and
    the score mapping: ``RangeLLMCriterion`` (continuous ``[0, 1]``) and
    ``OrdinalLLMCriterion`` (ordinal ``{1..5}``). ``LLMCriterion`` is the
    public router that picks an engine and delegates to it.

    Args:
        name: Criterion identifier.
        llm_client: LLM client for evaluation.
        prompt_template: Prompt template string with ``$variable`` placeholders.
        threshold: Minimum score to pass (continuous criteria); ordinal
            criteria run in score mode and leave this at ``0.0``.
        temperature: Sampling temperature; falls back to ``DEFAULT_TEMPERATURE``.
        max_tokens: Maximum tokens for response.
    """

    # Set by concrete engines (RangeLLMCriterion / OrdinalLLMCriterion). Left
    # unset on the base so a new engine that forgets to declare them fails
    # loudly (AttributeError) rather than silently inheriting continuous values.
    SCALE: ClassVar[CriterionScale]
    RESPONSE_MODEL: ClassVar[type[BaseModel]]
    CONFIG_CLS: ClassVar[type[BaseCriterionConfig]]
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.3

    def __init__(
        self,
        name: str,
        llm_client: Any,
        prompt_template: str,
        *,
        threshold: float = 0.0,
        temperature: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(name, threshold)
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.temperature = self.DEFAULT_TEMPERATURE if temperature is None else temperature
        self.max_tokens = max_tokens

    @property
    def scale(self) -> CriterionScale:
        return self.SCALE

    @staticmethod
    def _load_prompt_and_config(
        engine_cls: type[BaseLLMCriterion],
        name: str,
        prompts_dir: Path,
        language: str,
        temperature: float | None,
    ) -> tuple[str, float, float]:
        """Load the prompt template and config for a criterion.

        Shared by both the concrete engines and the router so the file/config
        loading lives in one place.

        Returns:
            Tuple of ``(prompt_template, threshold, effective_temperature)``.
            ``threshold`` is ``0.0`` for configs without one (ordinal).

        Raises:
            FileNotFoundError: If config.json or the prompt file is missing.
        """
        config = engine_cls.CONFIG_CLS.load(prompts_dir / name / "config.json")

        prompt_file = prompts_dir / name / language / "prompt.md"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        prompt_template = prompt_file.read_text(encoding="utf-8")

        base_temp = engine_cls.DEFAULT_TEMPERATURE if temperature is None else temperature
        effective_temp = config.temperature if config.temperature is not None else base_temp

        logger.debug("Loaded criterion '%s' for language '%s'", name, language)
        return prompt_template, getattr(config, "threshold", 0.0), effective_temp

    @classmethod
    def from_config(
        cls,
        name: str,
        prompts_dir: Path,
        language: str,
        llm_client: Any,
        *,
        temperature: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Self:
        """Load a concrete engine from prompt files and config.json.

        Args:
            name: Criterion name (e.g., "faithfulness", "emic_validity").
            prompts_dir: Base directory for criterion prompts.
            language: Language code (e.g., "pt", "en").
            llm_client: LLM client for evaluation.
            temperature: Default temperature (overridden by config if set).
            max_tokens: Maximum tokens for response.

        Returns:
            Configured engine instance of the calling subclass.
        """
        prompt_template, threshold, effective_temp = cls._load_prompt_and_config(
            cls, name, prompts_dir, language, temperature
        )
        return cls(
            name=name,
            llm_client=llm_client,
            prompt_template=prompt_template,
            threshold=threshold,
            temperature=effective_temp,
            max_tokens=max_tokens,
        )

    def _evaluate_impl(self, **kwargs: Any) -> CriterionScore:
        """Render the prompt, call the LLM, and map the response to a score."""
        prompt = _render_prompt(self.prompt_template, **kwargs)
        response = self.llm_client.generate_structured(
            prompt=prompt,
            response_model=self.RESPONSE_MODEL,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self._score_from_response(response)

    def evaluate(self, **kwargs: Any) -> CriterionScore:
        """Evaluate content, wrapping errors via ``_error_score``."""
        try:
            return self._evaluate_impl(**kwargs)
        except Exception as e:
            logger.warning("Criterion '%s' evaluation failed: %s", self.name, e)
            return self._error_score(str(e))

    @abstractmethod
    def _score_from_response(self, response: Any) -> CriterionScore:
        """Map a structured LLM response to a CriterionScore."""
        ...

    def _error_score(self, error: str) -> CriterionScore:
        """Build the CriterionScore used when evaluation raises.

        Continuous default; ordinal engines override to keep ``scale``.
        """
        return CriterionScore(
            score=None,
            threshold=self.threshold,
            rationale="",
            error=error,
        )


class RangeLLMCriterion(BaseLLMCriterion):
    """Continuous ``[0, 1]`` LLM criterion (the default engine).

    Inherits ``DEFAULT_TEMPERATURE = 0.3`` from the base.
    """

    SCALE: ClassVar[CriterionScale] = "continuous"
    RESPONSE_MODEL: ClassVar[type[BaseModel]] = RangeCriterionResponse
    CONFIG_CLS: ClassVar[type[BaseCriterionConfig]] = LLMCriterionConfig

    def _score_from_response(self, response: Any) -> CriterionScore:
        return CriterionScore(
            score=validate_score(response.score),
            threshold=self.threshold,
            rationale=response.rationale,
        )


class OrdinalLLMCriterion(BaseLLMCriterion):
    """Ordinal ``{1..5}`` LLM criterion engine.

    Produces a ``CriterionScore`` with ``scale="ordinal"`` and an integer
    ``ordinal_score``. The continuous ``threshold`` is unused (fixed at
    ``0.0``); the criterion runs in ``score`` mode.
    """

    SCALE: ClassVar[CriterionScale] = "ordinal"
    RESPONSE_MODEL: ClassVar[type[BaseModel]] = OrdinalCriterionResponse
    CONFIG_CLS: ClassVar[type[BaseCriterionConfig]] = OrdinalCriterionConfig
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.1

    def _score_from_response(self, response: Any) -> CriterionScore:
        return CriterionScore(
            ordinal_score=validate_ordinal_score(response.score, ORDINAL_MIN, ORDINAL_MAX),
            scale="ordinal",
            threshold=self.threshold,
            rationale=response.rationale,
        )

    def _error_score(self, error: str) -> CriterionScore:
        return CriterionScore(
            ordinal_score=None,
            scale="ordinal",
            threshold=self.threshold,
            rationale="",
            error=error,
        )


_ENGINES_BY_SCALE: dict[CriterionScale, type[BaseLLMCriterion]] = {
    "continuous": RangeLLMCriterion,
    "ordinal": OrdinalLLMCriterion,
}


def _engine_for_scale(scale: CriterionScale) -> type[BaseLLMCriterion]:
    """Resolve the engine class for a scale, defaulting to continuous."""
    try:
        return _ENGINES_BY_SCALE[scale]
    except KeyError:
        raise ValueError(f"Unknown criterion scale: {scale!r}") from None


class LLMCriterion(BaseLLMCriterion):
    """Public router that delegates to a Range (default) or Ordinal engine.

    Extends ``BaseLLMCriterion`` so it is a drop-in ``JudgeCriterion``, but
    construction selects a concrete engine by ``scale`` and all evaluation is
    delegated to it. Existing callers keep using ``LLMCriterion`` /
    ``LLMCriterion.from_config`` and transparently get the continuous engine;
    pass ``scale="ordinal"`` for the ordinal one.

    Args:
        name: Criterion identifier.
        llm_client: LLM client for evaluation.
        prompt_template: Prompt template string with ``$variable`` placeholders.
        threshold: Minimum score to pass (continuous only).
        temperature: Sampling temperature; falls back to the engine default.
        max_tokens: Maximum tokens for response.
        scale: ``"continuous"`` (default) or ``"ordinal"``.
    """

    def __init__(
        self,
        name: str,
        llm_client: Any,
        prompt_template: str,
        *,
        threshold: float = 0.0,
        temperature: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        scale: CriterionScale = "continuous",
    ) -> None:
        self._engine: BaseLLMCriterion = _engine_for_scale(scale)(
            name,
            llm_client,
            prompt_template,
            threshold=threshold,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        super().__init__(
            name,
            llm_client,
            prompt_template,
            threshold=self._engine.threshold,
            temperature=self._engine.temperature,
            max_tokens=max_tokens,
        )

    @property
    def scale(self) -> CriterionScale:
        """Derived from the delegate engine (single source of truth)."""
        return self._engine.scale

    @classmethod
    def from_config(
        cls,
        name: str,
        prompts_dir: Path,
        language: str,
        llm_client: Any,
        *,
        temperature: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        scale: CriterionScale = "continuous",
    ) -> LLMCriterion:
        """Load the router with the engine indicated by ``scale``.

        Reads ``config.json`` + ``prompt.md`` once (via the shared loader for
        the chosen engine) and constructs the router around the result.

        Returns:
            Configured ``LLMCriterion`` delegating to the selected engine.
        """
        engine_cls = _engine_for_scale(scale)
        prompt_template, threshold, effective_temp = cls._load_prompt_and_config(
            engine_cls, name, prompts_dir, language, temperature
        )
        return cls(
            name=name,
            llm_client=llm_client,
            prompt_template=prompt_template,
            threshold=threshold,
            temperature=effective_temp,
            max_tokens=max_tokens,
            scale=scale,
        )

    def _evaluate_impl(self, **kwargs: Any) -> CriterionScore:
        return self._engine._evaluate_impl(**kwargs)

    def evaluate(self, **kwargs: Any) -> CriterionScore:
        return self._engine.evaluate(**kwargs)

    def _score_from_response(self, response: Any) -> CriterionScore:
        return self._engine._score_from_response(response)
