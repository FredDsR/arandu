"""``AnswerJudge`` — 4-criterion LLM pipeline composition (spec §6.6).

All criteria run in ``score`` mode (none reject); the pipeline's role
is to attach verdicts to each :class:`AnswerRecord` for later analysis.

Note on scope: spec §6.3 also describes a deterministic
``offset_coverage`` variant alongside the LLM ``passage_coverage``.
The deterministic variant is intentionally **not** wired here:

- The thesis's research question is semantic ("did retrieval support a
  faithful tacit-knowledge answer?"), not extractive (literal-byte
  overlap with the gold chunk).
- The triple-arm retriever emits content in :attr:`RetrievedPassage.payload`
  with no ``(start_char, end_char)``; char-overlap would systematically
  zero it out, biasing the cross-arm comparison.
- The Phase C ``KC = correctness * faithfulness`` metric doesn't read
  it, so its only role would be downstream analysis — and a standalone
  script can recompute char-overlap from the persisted records if a
  robustness sanity check ever needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arandu.shared.judge import (
    BaseJudge,
    JudgePipeline,
    JudgeStage,
    JudgeStep,
    LLMCriterionFactory,
)

if TYPE_CHECKING:
    from arandu.shared.judge.schemas import JudgePipelineResult
    from arandu.shared.llm_client import LLMClient
    from arandu.shared.rag.judge_answers.settings import JudgeAnswersSettings


# Spec §6.6 — order matches the pipeline composition described there.
# All four LLM criteria load from ``prompts/judge/criteria/<name>/``.
_LLM_CRITERIA = (
    "passage_coverage",
    "abstention",
    "answer_correctness",
    "answer_faithfulness",
)

# Non-answerable items have no gold answer, so correctness / faithfulness /
# passage_coverage are undefined for them. Only ``abstention`` is meaningful
# (it decides TA vs FC in the analysis confusion matrix), so they are judged
# on that criterion alone. The abstention prompt reads only the system's
# ``abstained`` flag, answer text, and rationale — no gold needed.
_ABSTENTION_ONLY = ("abstention",)


class AnswerJudge(BaseJudge):
    """Drive the 4-criterion LLM judge pipeline over one :class:`AnswerRecord`.

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
        super().__init__()  # calls _build_pipeline()

    def _build_pipeline(self) -> JudgePipeline:
        """Assemble the 4 LLM criteria into one score-mode stage.

        All-in-one stage rather than four single-criterion stages because
        the judges don't gate each other (no filter mode); a single stage
        avoids redundant ``JudgePipelineResult`` machinery and gives the
        analysis stage one flat ``criterion_scores`` dict to consume.

        Also builds a sibling abstention-only pipeline (same stage name,
        so the analysis classifier finds the ``abstention`` score the same
        way) for non-answerable items, which have no gold answer.
        """
        self._abstention_pipeline = JudgePipeline(
            stages=[
                JudgeStage(
                    name="answer_judge",
                    step=JudgeStep(criteria=list(_ABSTENTION_ONLY), factory=self.factory),
                    mode="score",
                )
            ],
        )
        step = JudgeStep(criteria=list(_LLM_CRITERIA), factory=self.factory)
        return JudgePipeline(
            stages=[JudgeStage(name="answer_judge", step=step, mode="score")],
        )

    def evaluate_abstention_only(self, **kwargs: object) -> JudgePipelineResult:
        """Run only the ``abstention`` criterion (for non-answerable items).

        Args:
            **kwargs: Template parameters for the abstention prompt
                (``abstained``, ``answer_text``, ``rationale``).

        Returns:
            A :class:`JudgePipelineResult` whose single stage carries the
            ``abstention`` :class:`CriterionScore`.
        """
        return self._abstention_pipeline.evaluate(**kwargs)
