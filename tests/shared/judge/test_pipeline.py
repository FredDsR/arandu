"""Tests for JudgePipeline."""

from __future__ import annotations

from typing import Any

from arandu.shared.judge.pipeline import JudgePipeline, JudgeStage
from arandu.shared.judge.schemas import CriterionScore
from arandu.shared.judge.step import JudgeStep


class _StubCriterion:
    """Stub criterion that returns a fixed score."""

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


def _make_stage(
    name: str,
    score: float,
    threshold: float,
    mode: str = "score",
) -> JudgeStage:
    """Build a JudgeStage with a single stub criterion."""
    criterion = _StubCriterion(name, score, threshold=threshold)
    step = JudgeStep(criteria=[criterion])
    return JudgeStage(name=name, step=step, mode=mode)


class TestJudgePipeline:
    def test_all_filter_stages_pass(self) -> None:
        pipeline = JudgePipeline(
            stages=[
                _make_stage("s1", score=0.9, threshold=0.7, mode="filter"),
                _make_stage("s2", score=0.8, threshold=0.6, mode="filter"),
            ]
        )
        result = pipeline.evaluate()
        assert result.passed is True
        assert result.rejected_at is None
        assert "s1" in result.stage_results
        assert "s2" in result.stage_results

    def test_filter_failure_skips_subsequent_non_always(self) -> None:
        pipeline = JudgePipeline(
            stages=[
                _make_stage("gate", score=0.3, threshold=0.7, mode="filter"),
                _make_stage(
                    "skipped_score",
                    score=0.9,
                    threshold=0.5,
                    mode="score",
                ),
                _make_stage(
                    "skipped_filter",
                    score=0.9,
                    threshold=0.5,
                    mode="filter",
                ),
            ]
        )
        result = pipeline.evaluate()
        assert result.passed is False
        assert result.rejected_at == "gate"
        assert "gate" in result.stage_results
        assert "skipped_score" not in result.stage_results
        assert "skipped_filter" not in result.stage_results

    def test_score_mode_records_but_never_rejects(self) -> None:
        pipeline = JudgePipeline(
            stages=[
                _make_stage("low_score", score=0.1, threshold=0.9, mode="score"),
                _make_stage("next", score=0.8, threshold=0.5, mode="filter"),
            ]
        )
        result = pipeline.evaluate()
        assert result.passed is True
        assert result.rejected_at is None
        assert "low_score" in result.stage_results
        assert result.stage_results["low_score"].passed is False
        assert "next" in result.stage_results

    def test_always_mode_runs_after_rejection(self) -> None:
        pipeline = JudgePipeline(
            stages=[
                _make_stage("gate", score=0.2, threshold=0.5, mode="filter"),
                _make_stage("cleanup", score=1.0, threshold=0.0, mode="always"),
            ]
        )
        result = pipeline.evaluate()
        assert result.passed is False
        assert result.rejected_at == "gate"
        assert "cleanup" in result.stage_results

    def test_kwargs_forwarded_to_stages(self) -> None:
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
        stage = JudgeStage(name="cap", step=step, mode="score")

        pipeline = JudgePipeline(stages=[stage])
        pipeline.evaluate(context="ctx", question="q")
        assert criterion.received_kwargs == {
            "context": "ctx",
            "question": "q",
        }

    def test_empty_pipeline_passes(self) -> None:
        pipeline = JudgePipeline(stages=[])
        result = pipeline.evaluate()
        assert result.passed is True
        assert result.rejected_at is None
        assert result.stage_results == {}
