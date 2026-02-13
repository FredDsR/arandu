"""Tests for judge pipeline module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from gtranscriber.core.judge.pipeline import JudgePipeline
from gtranscriber.schemas import CriterionScore, ValidationScore

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_criteria(mocker: MockerFixture) -> list[Any]:
    """Create mock criteria."""
    criteria = []
    for name in ["faithfulness", "bloom_calibration", "informativeness"]:
        criterion = mocker.MagicMock()
        criterion.name = name
        criterion.evaluate.return_value = CriterionScore(
            criterion_name=name,
            score=0.8,
            rationale=f"{name} looks good",
        )
        criteria.append(criterion)
    return criteria


class TestJudgePipeline:
    """Tests for JudgePipeline class."""

    def test_initialization_with_weights(
        self,
        mock_criteria: list[Any],
    ) -> None:
        """Test pipeline initialization with custom weights."""
        weights = {
            "faithfulness": 0.4,
            "bloom_calibration": 0.3,
            "informativeness": 0.3,
        }

        pipeline = JudgePipeline(criteria=mock_criteria, weights=weights)

        assert pipeline.criteria == mock_criteria
        assert pipeline.weights == weights

    def test_initialization_with_equal_weights(
        self,
        mock_criteria: list[Any],
    ) -> None:
        """Test pipeline initialization defaults to equal weights."""
        pipeline = JudgePipeline(criteria=mock_criteria)

        # Should have equal weights
        assert all(w == pytest.approx(1.0 / 3) for w in pipeline.weights.values())

    def test_initialization_validates_weight_sum(
        self,
        mock_criteria: list[Any],
    ) -> None:
        """Test that weight sum validation catches invalid weights."""
        weights = {
            "faithfulness": 0.5,
            "bloom_calibration": 0.3,
            "informativeness": 0.1,  # Sum = 0.9, not 1.0
        }

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            JudgePipeline(criteria=mock_criteria, weights=weights)

    def test_initialization_validates_missing_criterion_weight(
        self,
        mock_criteria: list[Any],
    ) -> None:
        """Test that missing criterion weight raises error."""
        weights = {
            "faithfulness": 0.5,
            "bloom_calibration": 0.5,
            # Missing informativeness
        }

        with pytest.raises(ValueError, match="missing weights"):
            JudgePipeline(criteria=mock_criteria, weights=weights)

    def test_initialization_validates_extra_weights(
        self,
        mock_criteria: list[Any],
    ) -> None:
        """Test that extra weights raise error."""
        weights = {
            "faithfulness": 0.3,
            "bloom_calibration": 0.3,
            "informativeness": 0.3,
            "extra_criterion": 0.1,  # Not in criteria
        }

        with pytest.raises(ValueError, match="extra weights"):
            JudgePipeline(criteria=mock_criteria, weights=weights)

    def test_evaluate_calls_all_criteria(
        self,
        mock_criteria: list[Any],
    ) -> None:
        """Test that evaluate calls all criteria."""
        pipeline = JudgePipeline(criteria=mock_criteria)

        result = pipeline.evaluate(
            context="Test context",
            question="Test question?",
            answer="Test answer.",
        )

        # All criteria should be called
        for criterion in mock_criteria:
            criterion.evaluate.assert_called_once_with(
                context="Test context",
                question="Test question?",
                answer="Test answer.",
            )

        assert isinstance(result, ValidationScore)

    def test_evaluate_passes_extra_params(
        self,
        mock_criteria: list[Any],
    ) -> None:
        """Test that criterion-specific extra params are passed correctly."""
        pipeline = JudgePipeline(criteria=mock_criteria)

        pipeline.evaluate(
            context="Context",
            question="Q?",
            answer="A.",
            faithfulness={"extra": "param1"},
            bloom_calibration={"extra": "param2"},
        )

        # Check that extra params were passed to correct criteria
        mock_criteria[0].evaluate.assert_called_once()
        call_kwargs = mock_criteria[0].evaluate.call_args.kwargs
        assert call_kwargs.get("extra") == "param1"

        mock_criteria[1].evaluate.assert_called_once()
        call_kwargs = mock_criteria[1].evaluate.call_args.kwargs
        assert call_kwargs.get("extra") == "param2"

    def test_evaluate_calculates_overall_score(
        self,
        mock_criteria: list[Any],
    ) -> None:
        """Test that overall score is calculated correctly."""
        # Set specific scores for each criterion
        mock_criteria[0].evaluate.return_value = CriterionScore(
            criterion_name="faithfulness", score=1.0, rationale="Perfect"
        )
        mock_criteria[1].evaluate.return_value = CriterionScore(
            criterion_name="bloom_calibration", score=0.5, rationale="OK"
        )
        mock_criteria[2].evaluate.return_value = CriterionScore(
            criterion_name="informativeness", score=0.0, rationale="Poor"
        )

        weights = {
            "faithfulness": 0.5,
            "bloom_calibration": 0.3,
            "informativeness": 0.2,
        }

        pipeline = JudgePipeline(criteria=mock_criteria, weights=weights)

        result = pipeline.evaluate(
            context="Context",
            question="Q?",
            answer="A.",
        )

        # Expected: 1.0*0.5 + 0.5*0.3 + 0.0*0.2 = 0.65
        assert result.overall_score == pytest.approx(0.65)

    def test_evaluate_builds_validation_score(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that ValidationScore is built correctly with all fields."""
        # Create specific criteria for CEP
        criteria = []
        criterion_scores = {
            "faithfulness": 0.9,
            "bloom_calibration": 0.8,
            "informativeness": 0.7,
            "self_containedness": 0.95,
        }

        for name, score in criterion_scores.items():
            criterion = mocker.MagicMock()
            criterion.name = name
            criterion.evaluate.return_value = CriterionScore(
                criterion_name=name,
                score=score,
                rationale=f"{name} rationale",
                thinking=f"{name} thinking",
            )
            criteria.append(criterion)

        weights = {
            "faithfulness": 0.3,
            "bloom_calibration": 0.25,
            "informativeness": 0.25,
            "self_containedness": 0.2,
        }

        pipeline = JudgePipeline(criteria=criteria, weights=weights)

        result = pipeline.evaluate(
            context="Context",
            question="Q?",
            answer="A.",
        )

        # Check individual scores
        assert result.faithfulness == 0.9
        assert result.bloom_calibration == 0.8
        assert result.informativeness == 0.7
        assert result.self_containedness == 0.95

        # Check rationales are combined
        assert "faithfulness: faithfulness rationale" in result.judge_rationale
        assert "bloom_calibration: bloom_calibration rationale" in result.judge_rationale

        # Check thinking traces are combined
        assert "[faithfulness]" in result.judge_thinking
        assert "faithfulness thinking" in result.judge_thinking

        # Check criterion scores are stored
        assert result.criterion_scores is not None
        assert len(result.criterion_scores) == 4

    def test_evaluate_handles_missing_criteria(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test that missing criteria get default scores in ValidationScore."""
        # Only create faithfulness criterion
        criterion = mocker.MagicMock()
        criterion.name = "faithfulness"
        criterion.evaluate.return_value = CriterionScore(
            criterion_name="faithfulness",
            score=0.9,
            rationale="Good",
        )

        pipeline = JudgePipeline(
            criteria=[criterion],
            weights={"faithfulness": 1.0},
        )

        result = pipeline.evaluate(
            context="Context",
            question="Q?",
            answer="A.",
        )

        # Faithfulness should be present
        assert result.faithfulness == 0.9

        # Others should have defaults
        assert result.bloom_calibration == 0.5
        assert result.informativeness == 0.5
        assert result.self_containedness == 1.0

    def test_overall_score_clamped_to_valid_range(
        self,
        mock_criteria: list[Any],
    ) -> None:
        """Test that overall score is clamped to [0.0, 1.0]."""
        # This shouldn't happen with valid scores, but test the safety net
        pipeline = JudgePipeline(criteria=mock_criteria)

        # Manually test the calculation method
        criterion_scores = {
            "faithfulness": CriterionScore(
                criterion_name="faithfulness", score=1.0, rationale="Test"
            ),
            "bloom_calibration": CriterionScore(
                criterion_name="bloom_calibration", score=1.0, rationale="Test"
            ),
            "informativeness": CriterionScore(
                criterion_name="informativeness", score=1.0, rationale="Test"
            ),
        }

        overall = pipeline._calculate_overall_score(criterion_scores)

        # Should be clamped to [0.0, 1.0]
        assert 0.0 <= overall <= 1.0
