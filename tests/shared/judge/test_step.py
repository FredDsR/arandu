"""Tests for JudgeStep."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from arandu.shared.judge.schemas import CriterionScore
from arandu.shared.judge.step import JudgeStep


class _StubCriterion:
    """Stub criterion for testing."""

    def __init__(self, name: str, score: float, threshold: float = 0.0) -> None:
        self.name = name
        self.threshold = threshold
        self._score = score

    def evaluate(self, **kwargs: Any) -> CriterionScore:
        return CriterionScore(score=self._score, threshold=0.0, rationale=f"{self.name} stub")


class TestJudgeStep:
    def test_all_criteria_pass(self) -> None:
        step = JudgeStep(
            criteria=[
                _StubCriterion("a", 0.8, threshold=0.7),
                _StubCriterion("b", 0.9, threshold=0.6),
            ],
        )
        result = step.evaluate()
        assert result.passed is True
        assert result.criterion_scores["a"].score == 0.8
        assert result.criterion_scores["a"].threshold == 0.7
        assert result.criterion_scores["b"].threshold == 0.6

    def test_one_criterion_fails(self) -> None:
        step = JudgeStep(
            criteria=[
                _StubCriterion("a", 0.8, threshold=0.7),
                _StubCriterion("b", 0.3, threshold=0.6),
            ],
        )
        result = step.evaluate()
        assert result.passed is False
        assert result.criterion_scores["a"].passed is True
        assert result.criterion_scores["b"].passed is False

    def test_threshold_from_criterion(self) -> None:
        step = JudgeStep(
            criteria=[_StubCriterion("a", 0.5, threshold=0.9)],
        )
        result = step.evaluate()
        assert result.criterion_scores["a"].threshold == 0.9
        assert result.criterion_scores["a"].passed is False

    def test_kwargs_forwarded_to_criteria(self) -> None:
        class _CaptureCriterion:
            name = "capture"
            threshold = 0.5

            def __init__(self) -> None:
                self.received_kwargs: dict[str, Any] = {}

            def evaluate(self, **kwargs: Any) -> CriterionScore:
                self.received_kwargs = kwargs
                return CriterionScore(score=1.0, threshold=0.0, rationale="ok")

        criterion = _CaptureCriterion()
        step = JudgeStep(criteria=[criterion])
        step.evaluate(context="ctx", question="q", answer="a")
        assert criterion.received_kwargs == {
            "context": "ctx",
            "question": "q",
            "answer": "a",
        }

    def test_string_criteria_resolved_via_factory(self) -> None:
        stub = _StubCriterion("faith", 0.9, threshold=0.7)
        factory = MagicMock()
        factory.get_criterion.return_value = stub

        step = JudgeStep(criteria=["faith"], factory=factory)
        result = step.evaluate()

        factory.get_criterion.assert_called_once_with("faith")
        assert result.criterion_scores["faith"].score == 0.9
        assert result.criterion_scores["faith"].threshold == 0.7

    def test_string_without_factory_raises(self) -> None:
        with pytest.raises(ValueError, match="factory"):
            JudgeStep(criteria=["faith"])

    def test_mixed_string_and_object_criteria(self) -> None:
        stub_obj = _StubCriterion("a", 0.8, threshold=0.5)
        stub_str = _StubCriterion("b", 0.9, threshold=0.6)

        factory = MagicMock()
        factory.get_criterion.return_value = stub_str

        step = JudgeStep(criteria=[stub_obj, "b"], factory=factory)
        result = step.evaluate()

        factory.get_criterion.assert_called_once_with("b")
        assert result.criterion_scores["a"].score == 0.8
        assert result.criterion_scores["a"].threshold == 0.5
        assert result.criterion_scores["b"].score == 0.9
        assert result.criterion_scores["b"].threshold == 0.6
