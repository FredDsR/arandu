"""``AnswerJudge`` — gated answer-judging pipeline (spec §6.6).

Five stages composed as a cascade of single-responsibility gates, so
each LLM-backed scoring stage runs only when its data dependencies are
satisfied:

1. ``abstention`` (LLM, ``score``) — always recorded. The TA/TC/FA/FC
   signal the analysis stage reads, so it must survive every stage-skip.
2. ``answerability_gate`` (heuristic, ``filter``) — passes iff the item
   has a gold answer (``is_answerable=True``). A reject leaves only the
   abstention score in the result.
3. ``retrieval_scoring`` (LLM, ``score``) — ``passage_coverage``. A
   retrieval-quality lens: did the retrieved passages cover the gold
   answer? Runs for every answerable item (TC + FA); the system's
   answer text is not consulted, so abstention does not gate it.
4. ``commitment_gate`` (heuristic, ``filter``) — passes iff the system
   committed (``abstained=False``). Lives downstream of the
   answerability gate; reaching it implies the item is answerable.
5. ``answer_scoring`` (LLM, ``score``) — ``answer_correctness``,
   ``answer_faithfulness``. Need a committed answer + gold, so they run
   only for TC.

This split fixes a TC-only narrowing of ``passage_coverage`` that an
earlier draft introduced by lumping it with the answer-scoring criteria:
passage coverage depends only on retrieval + gold, not on commitment.

The whole pipeline still stops scoring ``answer_correctness`` /
``answer_faithfulness`` on abstained answers (where ``answer_text`` is
``None``) per spec §6.1/§6.2, and still lets non-answerable items be
judged on abstention alone (no gold needed).

Note on scope: spec §6.3 also describes a deterministic *byte-offset*
``offset_coverage`` variant alongside the LLM ``passage_coverage``. That
specific variant is intentionally **not** wired here:

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

A *different* deterministic criterion, :class:`SourceRecoveryCriterion`,
**is** wired into the retrieval-scoring stage: it is token-containment
against the gold ``context`` (not byte offsets), returns ``None`` for the
payload arms that the offset variant would have biased, and is reported
as a prose-arms-only diagnostic that does not feed ``KC``.
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
from arandu.shared.rag.judge_answers.heuristic import (
    AnswerabilityGateCriterion,
    CommitmentGateCriterion,
    SourceRecoveryCriterion,
)

if TYPE_CHECKING:
    from arandu.shared.llm_client import LLMClient
    from arandu.shared.rag.judge_answers.settings import JudgeAnswersSettings


# Stage 1: the abstention signal, always recorded.
_ABSTENTION_CRITERIA = ("abstention",)
# Stage 3: retrieval-quality lens, run for every answerable item (TC + FA).
_RETRIEVAL_CRITERIA = ("passage_coverage",)
# Stage 5: answer-text quality, run only for True-Commitment items (TC).
# All LLM criteria load from ``prompts/judge/criteria/<name>/``.
_ANSWER_CRITERIA = (
    "answer_correctness",
    "answer_faithfulness",
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
        """Assemble the cascaded answer-judging pipeline.

        Two heuristic gates partition the LLM-backed scoring stages on
        their actual data dependencies:

        - the answerability gate (filter) fronts both retrieval and
          answer scoring (both need a gold answer);
        - the commitment gate (filter) fronts answer scoring alone
          (only the answer-text criteria need a committed answer).

        The retrieval-scoring stage mixes the LLM ``passage_coverage``
        with the deterministic :class:`SourceRecoveryCriterion`
        (``JudgeStep`` accepts factory-resolved strings + criterion
        objects in one list). Stage names are distinct so the analysis
        stage finds scores by criterion name regardless of which stages
        were skipped.
        """
        abstention_step = JudgeStep(criteria=list(_ABSTENTION_CRITERIA), factory=self.factory)
        answerability_step = JudgeStep(criteria=[AnswerabilityGateCriterion()])
        retrieval_step = JudgeStep(
            criteria=[
                *_RETRIEVAL_CRITERIA,
                SourceRecoveryCriterion(language=self._settings.language),
            ],
            factory=self.factory,
        )
        commitment_step = JudgeStep(criteria=[CommitmentGateCriterion()])
        answer_step = JudgeStep(criteria=list(_ANSWER_CRITERIA), factory=self.factory)
        return JudgePipeline(
            stages=[
                JudgeStage(name="abstention", step=abstention_step, mode="score"),
                JudgeStage(name="answerability_gate", step=answerability_step, mode="filter"),
                JudgeStage(name="retrieval_scoring", step=retrieval_step, mode="score"),
                JudgeStage(name="commitment_gate", step=commitment_step, mode="filter"),
                JudgeStage(name="answer_scoring", step=answer_step, mode="score"),
            ],
        )
