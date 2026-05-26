"""Tests for TA/TC/FA/FC labelling (spec §8.1)."""

from __future__ import annotations

from arandu.shared.judge.schemas import (
    CriterionScore,
    JudgePipelineResult,
    JudgeStepResult,
)
from arandu.shared.rag.analysis.classifier import classify_record
from arandu.shared.rag.schemas import AnswerRecord


def _record(
    *,
    is_answerable: bool,
    abstained: bool,
    judge_abstention_score: float | None,
    threshold: float = 0.7,
) -> AnswerRecord:
    """Build a minimal judged AnswerRecord for classifier tests.

    ``judge_abstention_score=None`` simulates a judge-error row (the
    criterion ran but errored - score field is None).
    """
    validation = JudgePipelineResult(
        stage_results={
            "answer_judge": JudgeStepResult(
                criterion_scores={
                    "abstention": CriterionScore(
                        score=judge_abstention_score,
                        threshold=threshold,
                        rationale="r",
                    )
                }
            )
        },
        passed=True,
    )
    return AnswerRecord(
        qa_pair_id="src:chk:0",
        question="q",
        retriever_id="bm25",
        chunker_id="cep_4k",
        top_k=5,
        passages=[],
        elapsed_ms=1.0,
        is_answerable=is_answerable,
        answer_text=None if abstained else "answer",
        abstained=abstained,
        rationale="r",
        answerer_model="qwen3:14b",
        answerer_temperature=0.2,
        validation=validation,
    )


class TestClassifyRecord:
    def test_answerable_committed_is_tc(self) -> None:
        record = _record(is_answerable=True, abstained=False, judge_abstention_score=0.1)
        assert classify_record(record) == "TC"

    def test_answerable_both_say_abstain_is_fa(self) -> None:
        record = _record(is_answerable=True, abstained=True, judge_abstention_score=0.9)
        assert classify_record(record) == "FA"

    def test_nonanswerable_both_say_abstain_is_ta(self) -> None:
        record = _record(is_answerable=False, abstained=True, judge_abstention_score=0.9)
        assert classify_record(record) == "TA"

    def test_nonanswerable_committed_is_fc(self) -> None:
        record = _record(is_answerable=False, abstained=False, judge_abstention_score=0.1)
        assert classify_record(record) == "FC"

    def test_disagreement_treats_as_committed(self) -> None:
        """answerer abstained but judge says commit → committed for analysis."""
        record = _record(is_answerable=True, abstained=True, judge_abstention_score=0.1)
        assert classify_record(record) == "TC"

    def test_disagreement_other_direction_treats_as_committed(self) -> None:
        """answerer committed but judge says abstain → committed for analysis."""
        record = _record(is_answerable=False, abstained=False, judge_abstention_score=0.9)
        assert classify_record(record) == "FC"

    def test_judge_error_returns_unknown(self) -> None:
        record = _record(is_answerable=True, abstained=False, judge_abstention_score=None)
        assert classify_record(record) == "unknown"

    def test_unjudged_returns_unknown(self) -> None:
        record = _record(is_answerable=True, abstained=False, judge_abstention_score=0.1)
        record = record.model_copy(update={"validation": None})
        assert classify_record(record) == "unknown"

    def test_threshold_override_changes_label(self) -> None:
        """Override τ=0.05 → score 0.1 now counts as abstain, flipping TC → FA."""
        record = _record(is_answerable=True, abstained=True, judge_abstention_score=0.1)
        assert classify_record(record) == "TC"  # default τ=0.7
        assert classify_record(record, threshold_override=0.05) == "FA"
