"""Tests for JudgeStep."""

from __future__ import annotations

from typing import Any

from arandu.shared.judge.schemas import CriterionScore
from arandu.shared.judge.step import JudgeStep


class _StubCriterion:
    """Stub criterion for testing."""

    def __init__(self, name: str, score: float) -> None:
        self.name = name
        self._score = score

    def evaluate(self, **kwargs: Any) -> CriterionScore:
        return CriterionScore(score=self._score, threshold=0.0, rationale=f"{self.name} stub")


class TestJudgeStep:
    def test_all_criteria_pass(self) -> None:
        step = JudgeStep(
            criteria=[_StubCriterion("a", 0.8), _StubCriterion("b", 0.9)],
            thresholds={"a": 0.7, "b": 0.6},
        )
        result = step.evaluate()
        assert result.passed is True
        assert result.criterion_scores["a"].score == 0.8
        assert result.criterion_scores["a"].threshold == 0.7
        assert result.criterion_scores["b"].threshold == 0.6

    def test_one_criterion_fails(self) -> None:
        step = JudgeStep(
            criteria=[_StubCriterion("a", 0.8), _StubCriterion("b", 0.3)],
            thresholds={"a": 0.7, "b": 0.6},
        )
        result = step.evaluate()
        assert result.passed is False
        assert result.criterion_scores["a"].passed is True
        assert result.criterion_scores["b"].passed is False

    def test_threshold_injected_into_score(self) -> None:
        step = JudgeStep(criteria=[_StubCriterion("a", 0.5)], thresholds={"a": 0.9})
        result = step.evaluate()
        assert result.criterion_scores["a"].threshold == 0.9
        assert result.criterion_scores["a"].passed is False

    def test_kwargs_forwarded_to_criteria(self) -> None:
        class _CaptureCriterion:
            name = "capture"

            def __init__(self) -> None:
                self.received_kwargs: dict[str, Any] = {}

            def evaluate(self, **kwargs: Any) -> CriterionScore:
                self.received_kwargs = kwargs
                return CriterionScore(score=1.0, threshold=0.0, rationale="ok")

        criterion = _CaptureCriterion()
        step = JudgeStep(criteria=[criterion], thresholds={"capture": 0.5})
        step.evaluate(context="ctx", question="q", answer="a")
        assert criterion.received_kwargs == {"context": "ctx", "question": "q", "answer": "a"}

    def test_missing_threshold_defaults_to_zero(self) -> None:
        step = JudgeStep(criteria=[_StubCriterion("a", 0.1)], thresholds={})
        result = step.evaluate()
        assert result.criterion_scores["a"].threshold == 0.0
        assert result.criterion_scores["a"].passed is True
