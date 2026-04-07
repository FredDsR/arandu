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
from arandu.qa.schemas import QAPairCEP, QAPairValidated
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
    from arandu.shared.judge.schemas import JudgePipelineResult
    from arandu.shared.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Bloom level descriptions fallback file
_VALIDATION_PROMPTS_DIR = get_project_root() / "prompts" / "qa" / "cep" / "validation"


class QAJudge(BaseJudge):
    """Validate QA pairs using a composable shared judge pipeline.

    Replaces the legacy QAValidator with a single-stage filter pipeline
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
        remember_criteria = [
            "faithfulness",
            "bloom_calibration",
            "informativeness",
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
    ) -> QAPairValidated:
        """Validate a single QA pair.

        Args:
            qa_pair: QA pair to validate.
            context: Source context for grounding check.

        Returns:
            QAPairValidated with pipeline result and pass/fail status.
        """
        try:
            bloom_level_desc = self._get_bloom_level_desc(qa_pair.bloom_level)

            pipeline = (
                self._remember_pipeline if qa_pair.bloom_level == "remember" else self._pipeline
            )

            result: JudgePipelineResult = pipeline.evaluate(
                context=context,
                question=qa_pair.question,
                answer=qa_pair.answer,
                bloom_level=qa_pair.bloom_level,
                bloom_level_desc=bloom_level_desc,
            )

            return QAPairValidated(
                **qa_pair.model_dump(),
                validation=result,
                is_valid=result.passed,
            )

        except Exception as e:
            logger.warning(f"Validation failed for QA pair: {e}", exc_info=True)
            return QAPairValidated(
                **qa_pair.model_dump(),
                validation=None,
                is_valid=True,
            )

    def validate_batch(
        self,
        qa_pairs: list[QAPairCEP],
        context: str,
    ) -> list[QAPairValidated]:
        """Validate multiple QA pairs.

        Args:
            qa_pairs: List of QA pairs to validate.
            context: Source context.

        Returns:
            List of validated QA pairs.
        """
        return [self.validate(pair, context) for pair in qa_pairs]

    def _get_bloom_level_desc(self, bloom_level: str) -> str:
        """Get human-readable Bloom level description.

        Loads from the validation data file and caches on first access.

        Args:
            bloom_level: Bloom level name (e.g. "remember", "analyze").

        Returns:
            Description string, or the level name itself as fallback.
        """
        if not hasattr(self, "_bloom_level_cache"):
            lang = self.judge_config.language
            data_file = _VALIDATION_PROMPTS_DIR / lang / "data.json"
            if data_file.exists():
                with open(data_file, encoding="utf-8") as f:
                    self._bloom_level_cache: dict[str, str] = json.load(f).get("bloom_levels", {})
            else:
                self._bloom_level_cache = {}
        return self._bloom_level_cache.get(bloom_level, bloom_level)
