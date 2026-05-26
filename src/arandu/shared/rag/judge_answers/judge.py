"""``AnswerJudge`` — 4-criterion pipeline composition (spec §6.6).

All criteria run in ``score`` mode (none reject); the pipeline's role
is to attach verdicts to each :class:`AnswerRecord` for later analysis.
The deterministic ``offset_coverage`` heuristic criterion runs alongside
the four LLM criteria so analysis can compare semantic vs literal
retrieval-coverage signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arandu.qa.criteria import OffsetCoverageCriterion
from arandu.shared.judge import (
    BaseJudge,
    JudgePipeline,
    JudgeStage,
    JudgeStep,
    LLMCriterionFactory,
)

if TYPE_CHECKING:
    from arandu.shared.llm_client import LLMClient
    from arandu.shared.rag.judge_answers.settings import JudgeAnswersSettings


# Spec §6.6 — order matches the pipeline composition described there.
# All four LLM criteria load from ``prompts/judge/criteria/<name>/``;
# the heuristic ``offset_coverage`` is registered as a custom criterion
# alongside the LLM passage_coverage so analysis carries both signals.
_LLM_CRITERIA = (
    "passage_coverage",
    "abstention",
    "answer_correctness",
    "answer_faithfulness",
)


class AnswerJudge(BaseJudge):
    """Drive the 4-criterion judge pipeline over one :class:`AnswerRecord`.

    Attributes:
        factory: The :class:`LLMCriterionFactory` used to load LLM
            criteria from disk.
    """

    def __init__(self, llm_client: LLMClient, settings: JudgeAnswersSettings) -> None:
        """Construct from an existing LLM client + settings snapshot.

        Args:
            llm_client: Unified LLM client built from
                :class:`JudgeAnswersSettings` (provider/model/api_key/base_url).
            settings: Per-call configuration (language, temperature, …).
        """
        self.factory = LLMCriterionFactory(
            llm_client=llm_client,
            language=settings.language,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        self._settings = settings
        # Register the deterministic offset_coverage variant BEFORE pipeline
        # construction so JudgeStep can resolve it by name alongside the
        # LLM criteria.
        self.factory.register_custom_criterion(OffsetCoverageCriterion())
        super().__init__()  # calls _build_pipeline()

    def _build_pipeline(self) -> JudgePipeline:
        """Assemble the 4 LLM criteria + offset_coverage into one score-mode stage.

        All-in-one stage rather than four single-criterion stages because
        the judges don't gate each other (no filter mode); a single stage
        avoids redundant ``JudgePipelineResult`` machinery and gives the
        analysis stage one flat ``criterion_scores`` dict to consume.
        """
        criteria: list[str] = [*_LLM_CRITERIA, "offset_coverage"]
        step = JudgeStep(criteria=criteria, factory=self.factory)
        return JudgePipeline(
            stages=[JudgeStage(name="answer_judge", step=step, mode="score")],
        )
