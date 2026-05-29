"""Tests for ordinal-output judge criteria.

Covers the ordinal extension of the composable judge module: a criterion
type that emits an integer score in ``{1..5}`` plus a rationale, instead of
the continuous ``[0, 1]`` score produced by ``LLMCriterion``. The ordinal
score is the first consumer of this extension (emic validity, Phase D).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from arandu.shared.judge.criterion import (
    LLMCriterion,
    OrdinalCriterionResponse,
    OrdinalLLMCriterion,
    RangeCriterionResponse,
    RangeLLMCriterion,
)
from arandu.shared.judge.pipeline import JudgePipeline, JudgeStage
from arandu.shared.judge.schemas import CriterionScore, JudgePipelineResult
from arandu.shared.judge.step import JudgeStep
from arandu.utils.text import validate_ordinal_score

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    """Create a mock LLM client mirroring tests/shared/judge/test_criterion.py."""
    client = mocker.MagicMock()
    client.provider.value = "ollama"
    client.model_id = "test-model"
    return client


class TestOrdinalCriterionResponse:
    """The structured response model enforces the {1..5} range."""

    @pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
    def test_accepts_in_range(self, value: int) -> None:
        resp = OrdinalCriterionResponse(score=value, rationale="ok")
        assert resp.score == value

    @pytest.mark.parametrize("value", [0, 6, -1, 10])
    def test_rejects_out_of_range(self, value: int) -> None:
        with pytest.raises(ValidationError):
            OrdinalCriterionResponse(score=value, rationale="bad")

    def test_rejects_non_integer(self) -> None:
        # A float that is not a whole number must not silently truncate.
        with pytest.raises(ValidationError):
            OrdinalCriterionResponse(score=3.5, rationale="bad")


class TestValidateOrdinalScore:
    """The validate_ordinal_score helper coerces and clamps to {1..5}."""

    def test_passes_through_valid_int(self) -> None:
        assert validate_ordinal_score(4) == 4

    def test_coerces_whole_float(self) -> None:
        assert validate_ordinal_score(5.0) == 5

    def test_clamps_above_range(self) -> None:
        assert validate_ordinal_score(9) == 5

    def test_clamps_below_range(self) -> None:
        assert validate_ordinal_score(0) == 1

    def test_default_on_garbage(self) -> None:
        assert validate_ordinal_score("not_a_number") == 3

    def test_custom_default(self) -> None:
        assert validate_ordinal_score(None, default=2) == 2


class TestCriterionScoreOrdinal:
    """CriterionScore carries ordinal results alongside continuous ones."""

    def test_continuous_score_unchanged(self) -> None:
        cs = CriterionScore(score=0.8, threshold=0.7, rationale="good")
        assert cs.scale == "continuous"
        assert cs.passed is True
        assert cs.ordinal_score is None

    def test_ordinal_score_field(self) -> None:
        cs = CriterionScore(
            ordinal_score=4,
            scale="ordinal",
            threshold=0.0,
            rationale="mostly emic",
        )
        assert cs.scale == "ordinal"
        assert cs.ordinal_score == 4
        assert cs.score is None

    def test_ordinal_passed_when_evaluated(self) -> None:
        # An ordinal criterion runs in score mode; a successful evaluation
        # is "passed" regardless of the continuous threshold (the emic filter
        # threshold is applied downstream, not here).
        cs = CriterionScore(
            ordinal_score=1,
            scale="ordinal",
            threshold=0.0,
            rationale="distortion",
        )
        assert cs.passed is True

    def test_ordinal_not_passed_on_error(self) -> None:
        cs = CriterionScore(
            ordinal_score=None,
            scale="ordinal",
            threshold=0.0,
            rationale="",
            error="LLM call failed",
        )
        assert cs.passed is False

    def test_ordinal_json_round_trip(self) -> None:
        cs = CriterionScore(
            ordinal_score=3,
            scale="ordinal",
            threshold=0.0,
            rationale="mixed framing",
        )
        restored = CriterionScore.model_validate_json(cs.model_dump_json())
        assert restored.scale == "ordinal"
        assert restored.ordinal_score == 3
        assert restored.score is None


class TestOrdinalLLMCriterion:
    """OrdinalLLMCriterion produces ordinal CriterionScores via the LLM."""

    def test_evaluate_success(self, mock_llm_client: Any) -> None:
        mock_llm_client.generate_structured.return_value = OrdinalCriterionResponse(
            score=4, rationale="leve deslize de enquadramento"
        )
        criterion = OrdinalLLMCriterion(
            name="emic_validity",
            llm_client=mock_llm_client,
            prompt_template="Context: $context\nQuestion: $question\nAnswer: $answer\n",
            temperature=0.1,
            max_tokens=1024,
        )

        result = criterion.evaluate(
            context="o pescador tira o barco quando a água sobe",
            question="Como o pescador sabe que é hora de tirar o barco?",
            answer="Quando percebe a água subindo rápido.",
        )

        assert isinstance(result, CriterionScore)
        assert result.scale == "ordinal"
        assert result.ordinal_score == 4
        assert result.score is None
        assert result.rationale == "leve deslize de enquadramento"

        call_kwargs = mock_llm_client.generate_structured.call_args.kwargs
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["response_model"] is OrdinalCriterionResponse

    def test_evaluate_clamps_out_of_range_response(self, mock_llm_client: Any) -> None:
        # If structured output somehow yields an out-of-range int, the
        # criterion clamps rather than emitting an invalid ordinal.
        mock_llm_client.generate_structured.return_value = OrdinalCriterionResponse.model_construct(
            score=9, rationale="overflow"
        )
        criterion = OrdinalLLMCriterion(
            name="emic_validity",
            llm_client=mock_llm_client,
            prompt_template="$context $question $answer",
        )
        result = criterion.evaluate(context="c", question="q", answer="a")
        assert result.ordinal_score == 5

    def test_evaluate_error_returns_blank_ordinal(self, mock_llm_client: Any) -> None:
        mock_llm_client.generate_structured.side_effect = RuntimeError("boom")
        criterion = OrdinalLLMCriterion(
            name="emic_validity",
            llm_client=mock_llm_client,
            prompt_template="$context $question $answer",
        )
        result = criterion.evaluate(context="c", question="q", answer="a")
        assert result.ordinal_score is None
        assert result.scale == "ordinal"
        assert result.error is not None
        assert result.passed is False

    def test_from_config(self, mock_llm_client: Any, tmp_path: Path) -> None:
        base = tmp_path / "criteria"
        crit_dir = base / "emic_validity" / "pt"
        crit_dir.mkdir(parents=True)
        (crit_dir / "prompt.md").write_text("Modo antropólogo: $context $question $answer")
        (base / "emic_validity" / "config.json").write_text(json.dumps({"temperature": 0.1}))

        criterion = OrdinalLLMCriterion.from_config(
            name="emic_validity",
            prompts_dir=base,
            language="pt",
            llm_client=mock_llm_client,
        )
        assert criterion.name == "emic_validity"
        assert criterion.temperature == 0.1
        assert "Modo antropólogo" in criterion.prompt_template


class TestLLMCriterionRouter:
    """LLMCriterion routes to a Range (default) or Ordinal engine."""

    def test_default_routes_to_range_engine(self, mock_llm_client: Any) -> None:
        criterion = LLMCriterion(
            name="faithfulness",
            llm_client=mock_llm_client,
            prompt_template="$context",
            threshold=0.7,
        )
        assert criterion.scale == "continuous"
        assert isinstance(criterion._engine, RangeLLMCriterion)
        assert criterion.threshold == 0.7

    def test_scale_ordinal_routes_to_ordinal_engine(self, mock_llm_client: Any) -> None:
        criterion = LLMCriterion(
            name="emic_validity",
            llm_client=mock_llm_client,
            prompt_template="$context",
            scale="ordinal",
        )
        assert criterion.scale == "ordinal"
        assert isinstance(criterion._engine, OrdinalLLMCriterion)

    def test_unknown_scale_rejected(self, mock_llm_client: Any) -> None:
        with pytest.raises(ValueError, match="scale"):
            LLMCriterion(
                name="x",
                llm_client=mock_llm_client,
                prompt_template="$context",
                scale="nonsense",  # type: ignore[arg-type]
            )

    def test_continuous_evaluation_delegated(self, mock_llm_client: Any) -> None:
        mock_llm_client.generate_structured.return_value = RangeCriterionResponse(
            score=0.8, rationale="grounded"
        )
        criterion = LLMCriterion(
            name="faithfulness",
            llm_client=mock_llm_client,
            prompt_template="$context",
            threshold=0.7,
        )
        result = criterion.evaluate(context="c")
        assert result.scale == "continuous"
        assert result.score == 0.8
        assert (
            mock_llm_client.generate_structured.call_args.kwargs["response_model"]
            is RangeCriterionResponse
        )

    def test_ordinal_evaluation_delegated(self, mock_llm_client: Any) -> None:
        mock_llm_client.generate_structured.return_value = OrdinalCriterionResponse(
            score=2, rationale="reenquadramento"
        )
        criterion = LLMCriterion(
            name="emic_validity",
            llm_client=mock_llm_client,
            prompt_template="$context",
            scale="ordinal",
        )
        result = criterion.evaluate(context="c")
        assert result.scale == "ordinal"
        assert result.ordinal_score == 2
        assert (
            mock_llm_client.generate_structured.call_args.kwargs["response_model"]
            is OrdinalCriterionResponse
        )

    def test_from_config_routes_ordinal(self, mock_llm_client: Any, tmp_path: Path) -> None:
        base = tmp_path / "criteria"
        crit_dir = base / "emic_validity" / "pt"
        crit_dir.mkdir(parents=True)
        (crit_dir / "prompt.md").write_text("$context")
        (base / "emic_validity" / "config.json").write_text(json.dumps({"temperature": 0.1}))

        criterion = LLMCriterion.from_config(
            name="emic_validity",
            prompts_dir=base,
            language="pt",
            llm_client=mock_llm_client,
            scale="ordinal",
        )
        assert criterion.scale == "ordinal"
        assert isinstance(criterion._engine, OrdinalLLMCriterion)
        assert criterion.temperature == 0.1


class TestMixedPipeline:
    """Ordinal and continuous criteria coexist in one pipeline."""

    def test_ordinal_score_stage_after_continuous_filter(self, mock_llm_client: Any) -> None:
        # Continuous filter criterion (router → Range engine) that passes.
        cont = LLMCriterion(
            name="faithfulness",
            llm_client=mock_llm_client,
            prompt_template="$context $question $answer",
            threshold=0.6,
        )
        emic = OrdinalLLMCriterion(
            name="emic_validity",
            llm_client=mock_llm_client,
            prompt_template="$context $question $answer",
        )

        def _route(**kwargs: Any) -> Any:
            if kwargs["response_model"] is OrdinalCriterionResponse:
                return OrdinalCriterionResponse(score=2, rationale="reenquadramento")
            return RangeCriterionResponse(score=0.9, rationale="grounded")

        mock_llm_client.generate_structured.side_effect = _route

        pipeline = JudgePipeline(
            stages=[
                JudgeStage(name="canonical", step=JudgeStep([cont]), mode="filter"),
                JudgeStage(name="emic", step=JudgeStep([emic]), mode="score"),
            ]
        )

        result = pipeline.evaluate(context="c", question="q", answer="a")

        assert result.passed is True  # continuous filter passed
        emic_score = result.stage_results["emic"].criterion_scores["emic_validity"]
        assert emic_score.scale == "ordinal"
        assert emic_score.ordinal_score == 2

        # Pipeline result persists with the ordinal score intact.
        restored = JudgePipelineResult.model_validate_json(result.model_dump_json())
        restored_emic = restored.stage_results["emic"].criterion_scores["emic_validity"]
        assert restored_emic.ordinal_score == 2
        assert restored_emic.scale == "ordinal"
