"""QA Judge using shared judge pipeline.

Validates QA pairs for faithfulness, Bloom calibration, informativeness,
and self-containedness using a composable, multi-criterion judge pipeline
backed by the shared judge module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from arandu.qa.config import get_judge_config
from arandu.qa.schemas import QAPairCEP, QAPairValidated
from arandu.shared.judge import JudgeCriterionFactory, JudgePipeline, JudgeStage, JudgeStep

if TYPE_CHECKING:
    from arandu.qa.config import CEPConfig, JudgeConfig
    from arandu.shared.judge.schemas import CriterionScore, JudgePipelineResult
    from arandu.shared.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Bloom level descriptions fallback file
_VALIDATION_PROMPTS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "prompts" / "qa" / "cep" / "validation"
)


class QAJudge:
    """Validate QA pairs using a composable shared judge pipeline.

    Replaces the legacy QAValidator with a single-stage filter pipeline
    backed by ``JudgeCriterionFactory`` and ``JudgePipeline``.

    Evaluates each QA pair on four criteria:
    - Faithfulness: Is the answer grounded in the context?
    - Bloom Calibration: Does the question match the proposed cognitive level?
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
            judge_config: Judge pipeline configuration. If None, loads from env.
        """
        self.validator_client = validator_client
        self.cep_config = cep_config
        self.judge_config = judge_config or get_judge_config()

        self._pipeline = self._build_pipeline()

        logger.info(
            f"QAJudge initialized with {validator_client.provider.value}/"
            f"{validator_client.model_id}"
        )

    def _build_pipeline(self) -> JudgePipeline:
        """Build a single-stage filter pipeline for CEP validation.

        Returns:
            Configured JudgePipeline with one filter stage.
        """
        factory = JudgeCriterionFactory(
            llm_client=self.validator_client,
            language=self.judge_config.language,
            temperature=self.judge_config.temperature,
            max_tokens=self.judge_config.max_tokens,
        )

        criteria = factory.get_criteria("cep_validation")

        step = JudgeStep(
            criteria=criteria,
            thresholds=self.judge_config.thresholds,
        )

        stage = JudgeStage(name="cep_validation", step=step, mode="filter")

        return JudgePipeline(stages=[stage])

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

            result: JudgePipelineResult = self._pipeline.evaluate(
                context=context,
                question=qa_pair.question,
                answer=qa_pair.answer,
                bloom_level=qa_pair.bloom_level,
                bloom_level_desc=bloom_level_desc,
            )

            # Safety net: force self_containedness=1.0 for remember level
            if qa_pair.bloom_level == "remember":
                self._force_self_containedness(result)

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

    def _force_self_containedness(self, result: JudgePipelineResult) -> None:
        """Force self_containedness score to 1.0 for remember-level pairs.

        Remember-level questions are inherently self-contained because they
        ask for direct recall of facts stated in the text.

        Args:
            result: Pipeline result to modify in-place.
        """
        stage_result = result.stage_results.get("cep_validation")
        if stage_result is None:
            return

        sc: CriterionScore | None = stage_result.criterion_scores.get("self_containedness")
        if sc is not None:
            sc.score = 1.0

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
