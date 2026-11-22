"""Tests for judge schemas."""

from __future__ import annotations

from arandu.shared.judge.schemas import (
    CriterionScore,
    JudgePipelineResult,
    JudgeStepResult,
)


class TestCriterionScore:
    def test_passed_when_score_meets_threshold(self) -> None:
        cs = CriterionScore(score=0.8, threshold=0.7, rationale="Good")
        assert cs.passed is True

    def test_failed_when_score_below_threshold(self) -> None:
        cs = CriterionScore(score=0.5, threshold=0.7, rationale="Weak")
        assert cs.passed is False

    def test_passed_when_score_equals_threshold(self) -> None:
        cs = CriterionScore(score=0.7, threshold=0.7, rationale="Exact")
        assert cs.passed is True

    def test_thinking_optional(self) -> None:
        cs = CriterionScore(score=0.8, threshold=0.7, rationale="Good")
        assert cs.thinking is None
        cs_with = CriterionScore(
            score=0.8, threshold=0.7, rationale="Good", thinking="I thought..."
        )
        assert cs_with.thinking == "I thought..."

    def test_error_always_fails(self) -> None:
        cs = CriterionScore(score=None, threshold=0.5, rationale="", error="LLM error")
        assert cs.passed is False
        assert cs.score is None
        assert cs.error == "LLM error"

    def test_none_score_without_error_fails(self) -> None:
        cs = CriterionScore(score=None, threshold=0.0, rationale="")
        assert cs.passed is False


class TestJudgeStepResult:
    def test_passed_when_all_criteria_pass(self) -> None:
        result = JudgeStepResult(
            criterion_scores={
                "a": CriterionScore(score=0.8, threshold=0.7, rationale="Good"),
                "b": CriterionScore(score=0.9, threshold=0.6, rationale="Great"),
            }
        )
        assert result.passed is True

    def test_failed_when_any_criterion_fails(self) -> None:
        result = JudgeStepResult(
            criterion_scores={
                "a": CriterionScore(score=0.8, threshold=0.7, rationale="Good"),
                "b": CriterionScore(score=0.4, threshold=0.6, rationale="Weak"),
            }
        )
        assert result.passed is False


class TestJudgePipelineResult:
    def test_serialization_roundtrip(self) -> None:
        result = JudgePipelineResult(
            stage_results={
                "stage1": JudgeStepResult(
                    criterion_scores={
                        "f": CriterionScore(score=0.8, threshold=0.7, rationale="Good"),
                    }
                ),
            },
            passed=True,
            rejected_at=None,
        )
        data = result.model_dump()
        restored = JudgePipelineResult.model_validate(data)
        assert restored.passed is True
        assert restored.stage_results["stage1"].criterion_scores["f"].score == 0.8

    def test_rejected_result(self) -> None:
        result = JudgePipelineResult(stage_results={}, passed=False, rejected_at="heuristic_filter")
        assert result.passed is False
        assert result.rejected_at == "heuristic_filter"
