"""QA Judge using shared judge pipeline.

Validates QA pairs for faithfulness, Bloom calibration, informativeness,
and self-containedness using a composable, multi-criterion judge pipeline
backed by the shared judge module.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from arandu.qa.config import get_judge_config
from arandu.shared.judge import (
    BaseJudge,
    JudgePipeline,
    JudgeStage,
    JudgeStep,
    LLMCriterionFactory,
)
from arandu.utils.paths import get_project_root

if TYPE_CHECKING:
    from arandu.qa.config import CEPConfig, JudgeConfig
    from arandu.qa.schemas import QAPairCEP
    from arandu.shared.judge.schemas import JudgePipelineResult
    from arandu.shared.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Bloom level descriptions fallback file
_VALIDATION_PROMPTS_DIR = get_project_root() / "prompts" / "qa" / "cep" / "validation"


class QAJudge(BaseJudge):
    """Validate QA pairs using a composable shared judge pipeline.

    Uses a single-stage filter pipeline
    backed by ``LLMCriterionFactory`` and ``JudgePipeline``.

    Evaluates each QA pair on four criteria:
    - Faithfulness: Is the answer grounded in the context?
    - Bloom Calibration: Does the question match the proposed level?
    - Informativeness: Does the answer reveal non-obvious knowledge?
    - Self-Containedness: Is the question understandable without context?
    """

    def __init__(
        self,
        validator_client: LLMClient,
        cep_config: CEPConfig,
        judge_config: JudgeConfig | None = None,
    ) -> None:
        """Initialize judge with LLM client and configuration.

        Args:
            validator_client: LLM client for judge evaluation.
            cep_config: CEP configuration.
            judge_config: Judge pipeline configuration.
                If None, loads from env.
        """
        self.cep_config = cep_config
        self.judge_config = judge_config or get_judge_config()

        self._factory = LLMCriterionFactory(
            llm_client=validator_client,
            language=self.judge_config.language,
            temperature=self.judge_config.temperature,
            max_tokens=self.judge_config.max_tokens,
        )

        super().__init__()  # calls _build_pipeline() which uses self._factory

    def _build_pipeline(self) -> JudgePipeline:
        """Build evaluation pipelines for CEP validation.

        Creates two pipelines: one with all criteria (default) and one
        without self_containedness (for remember-level pairs, which are
        inherently self-contained).

        Returns:
            Default JudgePipeline with all four criteria.
        """
        all_criteria = [
            "faithfulness",
            "bloom_calibration",
            "informativeness",
            "self_containedness",
        ]
        # Remember pairs are factual recall by design (the 3/1/1/1 factual base),
        # so informativeness (which scores down trivial/explicit content) is a
        # conceptual mismatch that rejected ~55% of remember pairs. Judge them on
        # grounding + level only; self_containedness is auto-1.0 for remember.
        remember_criteria = [
            "faithfulness",
            "bloom_calibration",
        ]

        default_step = JudgeStep(criteria=all_criteria, factory=self._factory)
        remember_step = JudgeStep(criteria=remember_criteria, factory=self._factory)

        self._remember_pipeline = JudgePipeline(
            stages=[JudgeStage(name="cep_validation", step=remember_step, mode="filter")]
        )

        return JudgePipeline(
            stages=[JudgeStage(name="cep_validation", step=default_step, mode="filter")]
        )

    def validate(
        self,
        qa_pair: QAPairCEP,
        context: str,
    ) -> QAPairCEP:
        """Validate a single QA pair.

        Args:
            qa_pair: QA pair to validate.
            context: Source context for grounding check.

        Returns:
            QAPairCEP with the ``validation`` field populated. ``is_valid``
            is derived automatically from ``validation.passed``.
        """
        try:
            bloom_level_desc = self._get_bloom_level_desc(qa_pair.bloom_level)
            bloom_ladder = self._get_bloom_ladder()

            pipeline = (
                self._remember_pipeline if qa_pair.bloom_level == "remember" else self._pipeline
            )

            result: JudgePipelineResult = pipeline.evaluate(
                context=context,
                question=qa_pair.question,
                answer=qa_pair.answer,
                bloom_level=qa_pair.bloom_level,
                bloom_level_desc=bloom_level_desc,
                bloom_ladder=bloom_ladder,
            )

            return qa_pair.model_copy(update={"validation": result})

        except Exception as e:
            logger.warning(f"Validation failed for QA pair: {e}", exc_info=True)
            # Return the original pair unchanged — no verdict, is_valid stays None.
            return qa_pair

    def validate_batch(
        self,
        qa_pairs: list[QAPairCEP],
        context: str,
    ) -> list[QAPairCEP]:
        """Validate multiple QA pairs.

        Args:
            qa_pairs: List of QA pairs to validate.
            context: Source context.

        Returns:
            List of QA pairs with their ``validation`` field populated.
        """
        return [self.validate(pair, context) for pair in qa_pairs]

    def _bloom_levels(self) -> dict[str, str]:
        """Load and cache the Bloom-level descriptions from the validation data file.

        Returns:
            Mapping of Bloom level name to its reference description (empty if the
            data file is missing).
        """
        if not hasattr(self, "_bloom_level_cache"):
            lang = self.judge_config.language
            data_file = _VALIDATION_PROMPTS_DIR / lang / "data.json"
            if data_file.exists():
                with open(data_file, encoding="utf-8") as f:
                    self._bloom_level_cache: dict[str, str] = json.load(f).get("bloom_levels", {})
            else:
                self._bloom_level_cache = {}
        return self._bloom_level_cache

    def _get_bloom_level_desc(self, bloom_level: str) -> str:
        """Get the human-readable description of a single Bloom level.

        Args:
            bloom_level: Bloom level name (e.g. "remember", "analyze").

        Returns:
            Description string, or the level name itself as fallback.
        """
        return self._bloom_levels().get(bloom_level, bloom_level)

    def _get_bloom_ladder(self) -> str:
        """Format all Bloom-level definitions as a reference ladder.

        Gives the bloom_calibration judge the full taxonomy (not just the declared
        level) so it can identify the level a question actually requires against our
        definitions rather than its own training prior.

        Returns:
            One ``- <level>: <description>`` line per level, or empty string if no
            descriptions are available.
        """
        return "\n".join(f"- {name}: {desc}" for name, desc in self._bloom_levels().items())
