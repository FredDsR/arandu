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

    def __init__(
        self,
        llm_client: Any,
        pipeline: JudgePipeline,
        **kwargs: Any,
    ) -> None:
        self._test_pipeline = pipeline
        super().__init__(llm_client=llm_client, **kwargs)

    def _build_pipeline(self) -> JudgePipeline:
        return self._test_pipeline


def _make_mock_llm_client() -> MagicMock:
    """Create a mock LLMClient with required attributes."""
    client = MagicMock()
    client.provider.value = "ollama"
    client.model_id = "llama3.1:8b"
    return client


class TestBaseJudge:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            BaseJudge(  # type: ignore[abstract]
                llm_client=_make_mock_llm_client(),
            )

    def test_concrete_subclass_works(self) -> None:
        criterion = _StubCriterion("a", 0.9, threshold=0.5)
        step = JudgeStep(criteria=[criterion])
        stage = JudgeStage(name="s", step=step, mode="filter")
        pipeline = JudgePipeline(stages=[stage])
        mock_client = _make_mock_llm_client()

        judge = _ConcreteJudge(llm_client=mock_client, pipeline=pipeline)

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
        mock_client = _make_mock_llm_client()

        judge = _ConcreteJudge(llm_client=mock_client, pipeline=mock_pipeline)
        result = judge.evaluate(context="ctx", question="q")

        mock_pipeline.evaluate.assert_called_once_with(context="ctx", question="q")
        assert result is expected_result

    def test_factory_is_created(self) -> None:
        """Test that BaseJudge creates a JudgeCriterionFactory."""
        mock_pipeline = MagicMock(spec=JudgePipeline)
        mock_client = _make_mock_llm_client()

        judge = _ConcreteJudge(llm_client=mock_client, pipeline=mock_pipeline)

        assert judge._factory is not None
        assert judge._factory.llm_client is mock_client
        assert judge._factory.language == "pt"

    def test_custom_params_passed_to_factory(self) -> None:
        """Test that language/temperature/max_tokens reach the factory."""
        mock_pipeline = MagicMock(spec=JudgePipeline)
        mock_client = _make_mock_llm_client()

        judge = _ConcreteJudge(
            llm_client=mock_client,
            pipeline=mock_pipeline,
            language="en",
            temperature=0.5,
            max_tokens=4096,
        )

        assert judge._factory.language == "en"
        assert judge._factory.temperature == 0.5
        assert judge._factory.max_tokens == 4096

    def test_init_logs_class_name(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that initialization logs the class name and model info."""
        mock_pipeline = MagicMock(spec=JudgePipeline)
        mock_client = _make_mock_llm_client()

        with caplog.at_level(logging.INFO, logger="arandu.shared.judge.judge"):
            _ConcreteJudge(llm_client=mock_client, pipeline=mock_pipeline)

        assert "_ConcreteJudge" in caplog.text
        assert "ollama/llama3.1:8b" in caplog.text
