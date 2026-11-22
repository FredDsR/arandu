"""Tests for BaseJudge ABC."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from arandu.shared.judge.judge import BaseJudge
from arandu.shared.judge.pipeline import JudgePipeline, JudgeStage
from arandu.shared.judge.schemas import (
    CriterionScore,
    JudgePipelineResult,
    JudgeStepResult,
)
from arandu.shared.judge.step import JudgeStep


class _StubCriterion:
    """Stub criterion for testing."""

    def __init__(self, name: str, score: float, threshold: float = 0.0) -> None:
        self.name = name
        self.threshold = threshold
        self._score = score

    def evaluate(self, **kwargs: Any) -> CriterionScore:
        return CriterionScore(
            score=self._score,
            threshold=0.0,
            rationale=f"{self.name} stub",
        )


class _ConcreteJudge(BaseJudge):
    """Concrete subclass of BaseJudge for testing."""

    def __init__(self, pipeline: JudgePipeline) -> None:
        self._test_pipeline = pipeline
        super().__init__()

    def _build_pipeline(self) -> JudgePipeline:
        return self._test_pipeline


class TestBaseJudge:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            BaseJudge()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        criterion = _StubCriterion("a", 0.9, threshold=0.5)
        step = JudgeStep(criteria=[criterion])
        stage = JudgeStage(name="s", step=step, mode="filter")
        pipeline = JudgePipeline(stages=[stage])

        judge = _ConcreteJudge(pipeline=pipeline)

        assert judge._pipeline is pipeline

    def test_evaluate_delegates_to_pipeline(self) -> None:
        mock_pipeline = MagicMock(spec=JudgePipeline)
        expected_result = JudgePipelineResult(
            stage_results={
                "s": JudgeStepResult(
                    criterion_scores={
                        "a": CriterionScore(
                            score=0.9,
                            threshold=0.5,
                            rationale="ok",
                        )
                    }
                )
            },
            passed=True,
        )
        mock_pipeline.evaluate.return_value = expected_result

        judge = _ConcreteJudge(pipeline=mock_pipeline)
        result = judge.evaluate(context="ctx", question="q")

        mock_pipeline.evaluate.assert_called_once_with(context="ctx", question="q")
        assert result is expected_result

    def test_init_logs_class_name(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that initialization logs the class name."""
        mock_pipeline = MagicMock(spec=JudgePipeline)

        with caplog.at_level(logging.INFO, logger="arandu.shared.judge.judge"):
            _ConcreteJudge(pipeline=mock_pipeline)

        assert "_ConcreteJudge" in caplog.text
