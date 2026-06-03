"""Tests for the gated AnswerJudge pipeline (spec §6.6).

Mocks the LLMClient so the real pipeline structure runs (abstention ->
answerability gate -> retrieval scoring -> commitment gate -> answer
scoring) without LLM calls. Verifies the cascaded gates produce the
expected per-quadrant criterion coverage.
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
        "retrieved_text": "Maria mora em Itaqui",
        "passages_are_payload": False,
        "question": "Onde Maria mora?",
        "gold_answer": "Em Itaqui.",
        "context": "Maria mora em Itaqui na beira do rio",
    }


class TestAnswerJudgePipeline:
    """The cascaded gates partition criteria by their data dependencies."""

    def test_true_commitment_runs_all_stages(self) -> None:
        judge, _ = _judge()
        result = judge.evaluate(**_kwargs(is_answerable=True, abstained="false"))
        assert result.passed is True
        # All five stages recorded; both LLM-scoring stages populated.
        assert set(result.stage_results) == {
            "abstention",
            "answerability_gate",
            "retrieval_scoring",
            "commitment_gate",
            "answer_scoring",
        }
        retrieval = result.stage_results["retrieval_scoring"].criterion_scores
        assert "passage_coverage" in retrieval
        # Deterministic source_recovery rides alongside the LLM criterion
        # in the same stage; here the retrieved text is fully contained.
        assert "source_recovery" in retrieval
        assert retrieval["source_recovery"].score == 1.0
        assert set(result.stage_results["answer_scoring"].criterion_scores) == {
            "answer_correctness",
            "answer_faithfulness",
        }

    def test_false_abstention_keeps_passage_coverage(self) -> None:
        # Answerable + abstained (FA): retrieval can still be evaluated
        # against the gold answer; only the answer-text scoring stage skips.
        judge, _ = _judge()
        result = judge.evaluate(**_kwargs(is_answerable=True, abstained="true"))
        assert result.passed is False
        assert result.rejected_at == "commitment_gate"
        assert "retrieval_scoring" in result.stage_results
        retrieval = result.stage_results["retrieval_scoring"].criterion_scores
        # Both retrieval-stage criteria run for FA, not just passage_coverage.
        assert "passage_coverage" in retrieval
        assert "source_recovery" in retrieval
        assert retrieval["source_recovery"].score == 1.0
        assert "answer_scoring" not in result.stage_results

    def test_nonanswerable_committed_only_abstention(self) -> None:
        judge, _ = _judge()
        result = judge.evaluate(**_kwargs(is_answerable=False, abstained="false"))
        assert result.passed is False
        assert result.rejected_at == "answerability_gate"
        assert "abstention" in result.stage_results
        # Both downstream LLM stages skipped (no gold).
        assert "retrieval_scoring" not in result.stage_results
        assert "answer_scoring" not in result.stage_results

    def test_nonanswerable_abstained_only_abstention(self) -> None:
        judge, _ = _judge()
        result = judge.evaluate(**_kwargs(is_answerable=False, abstained="true"))
        assert result.passed is False
        assert result.rejected_at == "answerability_gate"
        assert "retrieval_scoring" not in result.stage_results
        assert "answer_scoring" not in result.stage_results

    def test_non_tc_items_skip_answer_scoring_llm_calls(self) -> None:
        # FA: 1 LLM call for abstention + 1 for passage_coverage = 2.
        # Non-answerable (TA/FC): 1 LLM call for abstention only.
        judge, llm = _judge()
        judge.evaluate(**_kwargs(is_answerable=True, abstained="true"))
        assert llm.generate_structured.call_count == 2
        llm.reset_mock()
        judge.evaluate(**_kwargs(is_answerable=False, abstained="true"))
        assert llm.generate_structured.call_count == 1
