"""Tests for the gated AnswerJudge pipeline (spec §6.6).

Mocks the LLMClient so the real pipeline structure runs (abstention ->
commitment gate -> gold scoring) without LLM calls. Verifies the
commitment gate short-circuits the gold-scoring stage for everything
except a True-Commitment candidate.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from arandu.shared.judge.criterion import CriterionResponse
from arandu.shared.rag.judge_answers.judge import AnswerJudge
from arandu.shared.rag.judge_answers.settings import JudgeAnswersSettings


def _judge() -> tuple[AnswerJudge, MagicMock]:
    """Build an AnswerJudge whose LLM criteria all return a fixed score."""
    llm = MagicMock()
    llm.generate_structured.return_value = CriterionResponse(score=0.8, rationale="ok")
    judge = AnswerJudge(llm_client=llm, settings=JudgeAnswersSettings(provider="ollama"))
    return judge, llm


def _kwargs(*, is_answerable: bool, abstained: str) -> dict[str, object]:
    return {
        "is_answerable": is_answerable,
        "abstained": abstained,
        "answer_text": "Em Itaqui." if abstained == "false" else "",
        "system_answer": "Em Itaqui." if abstained == "false" else "",
        "rationale": "r",
        "passages_text": "p",
        "question": "Onde Maria mora?",
        "gold_answer": "Em Itaqui.",
        "context": "Maria mora em Itaqui.",
        "bloom_level": "remember",
        "question_type": "factual",
        "reasoning_trace": "",
        "tacit_inference": "",
    }


class TestAnswerJudgePipeline:
    """The commitment gate gates the gold-scoring stage."""

    def test_true_commitment_runs_gold_scoring(self) -> None:
        judge, _ = _judge()
        result = judge.evaluate(**_kwargs(is_answerable=True, abstained="false"))
        assert result.passed is True
        assert "abstention" in result.stage_results
        assert "gold_scoring" in result.stage_results
        gold = result.stage_results["gold_scoring"].criterion_scores
        assert set(gold) == {"answer_correctness", "answer_faithfulness", "passage_coverage"}
        # abstention is always recorded, in its own stage.
        assert "abstention" in result.stage_results["abstention"].criterion_scores

    def test_answerable_abstained_skips_gold(self) -> None:
        judge, _ = _judge()
        result = judge.evaluate(**_kwargs(is_answerable=True, abstained="true"))
        assert result.passed is False
        assert result.rejected_at == "commitment_gate"
        assert "abstention" in result.stage_results
        assert "gold_scoring" not in result.stage_results

    def test_nonanswerable_committed_skips_gold(self) -> None:
        judge, _ = _judge()
        result = judge.evaluate(**_kwargs(is_answerable=False, abstained="false"))
        assert result.passed is False
        assert "gold_scoring" not in result.stage_results
        assert "abstention" in result.stage_results["abstention"].criterion_scores

    def test_nonanswerable_abstained_skips_gold(self) -> None:
        judge, _ = _judge()
        result = judge.evaluate(**_kwargs(is_answerable=False, abstained="true"))
        assert result.passed is False
        assert "gold_scoring" not in result.stage_results

    def test_gate_does_not_call_llm(self) -> None:
        # On a rejected (non-TC) item only the abstention criterion hits the
        # LLM; the gate is heuristic and gold scoring is skipped -> exactly
        # one generate_structured call.
        judge, llm = _judge()
        judge.evaluate(**_kwargs(is_answerable=False, abstained="true"))
        assert llm.generate_structured.call_count == 1
