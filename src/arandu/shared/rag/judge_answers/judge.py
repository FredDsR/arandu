"""``AnswerJudge`` — gated answer-judging pipeline (spec §6.6).

Three stages, run in order:

1. ``abstention`` (LLM, ``score`` mode) — always recorded. This is the
   TA/TC/FA/FC signal the analysis stage reads, so it must survive even
   when later stages are skipped.
2. ``commitment_gate`` (heuristic, ``filter`` mode) — passes only for a
   True-Commitment candidate (answerable + committed). A reject
   short-circuits the gold-scoring stage. Deterministic, no LLM call.
3. ``gold_scoring`` (LLM, ``score`` mode) — ``answer_correctness``,
   ``answer_faithfulness``, ``passage_coverage``. These need a gold
   answer + a committed answer, so they run only for TC items.

This gating replaces the earlier "run all four criteria unconditionally"
composition: it stops scoring correctness/faithfulness on abstained
answers (where ``answer_text`` is None) per spec §6.1/§6.2, and lets
non-answerable items be judged on abstention alone (no gold needed).

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
from arandu.shared.rag.judge_answers.heuristic import CommitmentGateCriterion

if TYPE_CHECKING:
    from arandu.shared.llm_client import LLMClient
    from arandu.shared.rag.judge_answers.settings import JudgeAnswersSettings


# Stage 1: the abstention signal, always recorded.
_ABSTENTION_CRITERIA = ("abstention",)
# Stage 3: gold-requiring criteria, run only for True-Commitment items.
# All load from ``prompts/judge/criteria/<name>/``.
_GOLD_CRITERIA = (
    "answer_correctness",
    "answer_faithfulness",
    "passage_coverage",
)


class AnswerJudge(BaseJudge):
    """Drive the gated answer-judging pipeline over one :class:`AnswerRecord`.

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
        """Assemble the abstention -> commitment-gate -> gold-scoring pipeline.

        The commitment gate (heuristic, filter mode) sits between the
        always-recorded abstention stage and the gold-requiring stage, so
        correctness/faithfulness/passage_coverage run only for True-
        Commitment items. Stage names are distinct so the analysis stage
        still finds the ``abstention`` score by criterion name.
        """
        abstention_step = JudgeStep(criteria=list(_ABSTENTION_CRITERIA), factory=self.factory)
        gate_step = JudgeStep(criteria=[CommitmentGateCriterion()])
        gold_step = JudgeStep(criteria=list(_GOLD_CRITERIA), factory=self.factory)
        return JudgePipeline(
            stages=[
                JudgeStage(name="abstention", step=abstention_step, mode="score"),
                JudgeStage(name="commitment_gate", step=gate_step, mode="filter"),
                JudgeStage(name="gold_scoring", step=gold_step, mode="score"),
            ],
        )
