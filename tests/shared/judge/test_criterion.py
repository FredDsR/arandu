"""Tests for shared judge criterion module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from arandu.shared.judge.criterion import CriterionResponse, HeuristicCriterion, LLMCriterion
from arandu.shared.judge.schemas import CriterionScore
from arandu.shared.llm_client import StructuredOutputError

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    """Create a mock LLM client."""
    client = mocker.MagicMock()
    client.provider.value = "ollama"
    client.model_id = "test-model"
    return client


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Create temporary prompts directory structure."""
    base_dir = tmp_path / "criteria"
    faithfulness_dir = base_dir / "faithfulness" / "pt"
    faithfulness_dir.mkdir(parents=True)

    # Create prompt template file (rubric is already inlined)
    prompt_file = faithfulness_dir / "prompt.md"
    prompt_file.write_text(
        "Context: $context\nQuestion: $question\nAnswer: $answer\nRubric content here\n"
    )

    # Create config.json at criterion level (not per-language)
    config_file = base_dir / "faithfulness" / "config.json"
    config_file.write_text(json.dumps({"threshold": 0.7}))

    return base_dir


class TestLLMCriterion:
    """Tests for LLMCriterion class."""

    def test_initialization(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test criterion initialization loads files."""
        criterion = LLMCriterion(
            name="faithfulness",
            prompts_dir=prompts_dir,
            language="pt",
            llm_client=mock_llm_client,
        )

        assert criterion.name == "faithfulness"
        assert criterion.language == "pt"
        assert criterion.llm_client == mock_llm_client
        assert "Rubric content here" in criterion.prompt_template
        assert "$context" in criterion.prompt_template

    def test_threshold_loaded_from_config(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test that threshold is loaded from config.json."""
        criterion = LLMCriterion(
            name="faithfulness",
            prompts_dir=prompts_dir,
            language="pt",
            llm_client=mock_llm_client,
        )
        assert criterion.threshold == 0.7

    def test_initialization_missing_prompt(
        self,
        mock_llm_client: Any,
        tmp_path: Path,
    ) -> None:
        """Test that FileNotFoundError is raised for missing prompt."""
        criterion_dir = tmp_path / "criteria" / "test" / "pt"
        criterion_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Prompt file not found"):
            LLMCriterion(
                name="test",
                prompts_dir=tmp_path / "criteria",
                language="pt",
                llm_client=mock_llm_client,
            )

    def test_missing_config_json_raises(
        self,
        mock_llm_client: Any,
        tmp_path: Path,
    ) -> None:
        """Test that missing config.json raises FileNotFoundError."""
        criterion_dir = tmp_path / "criteria" / "test" / "pt"
        criterion_dir.mkdir(parents=True)
        (criterion_dir / "prompt.md").write_text("Prompt")

        with pytest.raises(FileNotFoundError, match=r"config\.json"):
            LLMCriterion(
                name="test",
                prompts_dir=tmp_path / "criteria",
                language="pt",
                llm_client=mock_llm_client,
            )

    def test_missing_threshold_key_raises(
        self,
        mock_llm_client: Any,
        tmp_path: Path,
    ) -> None:
        """Test that missing threshold key in config.json raises KeyError."""
        criterion_dir = tmp_path / "criteria" / "test" / "pt"
        criterion_dir.mkdir(parents=True)
        (criterion_dir / "prompt.md").write_text("Prompt")
        config_file = tmp_path / "criteria" / "test" / "config.json"
        config_file.write_text(json.dumps({"other_key": 42}))

        with pytest.raises(KeyError, match="threshold"):
            LLMCriterion(
                name="test",
                prompts_dir=tmp_path / "criteria",
                language="pt",
                llm_client=mock_llm_client,
            )

    def test_evaluate_success(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test successful evaluation."""
        mock_llm_client.generate_structured.return_value = CriterionResponse(
            score=0.8, rationale="Good quality"
        )

        criterion = LLMCriterion(
            name="faithfulness",
            prompts_dir=prompts_dir,
            language="pt",
            llm_client=mock_llm_client,
            temperature=0.3,
            max_tokens=1024,
        )

        result = criterion.evaluate(
            context="Test context",
            question="Test question?",
            answer="Test answer.",
        )

        assert isinstance(result, CriterionScore)
        assert result.score == 0.8
        assert result.threshold == 0.7
        assert result.rationale == "Good quality"

        # Verify LLM was called with correct params
        mock_llm_client.generate_structured.assert_called_once()
        call_kwargs = mock_llm_client.generate_structured.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["response_model"] is CriterionResponse

    def test_evaluate_builds_prompt_with_kwargs(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test that evaluate passes kwargs to prompt building."""
        mock_llm_client.generate_structured.return_value = CriterionResponse(
            score=0.7, rationale="Decent"
        )

        criterion = LLMCriterion(
            name="faithfulness",
            prompts_dir=prompts_dir,
            language="pt",
            llm_client=mock_llm_client,
        )

        criterion.evaluate(
            context="Context",
            question="Q?",
            answer="A.",
        )

        # Check that prompt was built with kwargs
        call_args = mock_llm_client.generate_structured.call_args
        prompt = call_args.kwargs["prompt"]
        assert "Context: Context" in prompt
        assert "Question: Q?" in prompt
        assert "Answer: A." in prompt
        assert "Rubric content here" in prompt

    def test_evaluate_handles_structured_output_error(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test that StructuredOutputError returns neutral score."""
        mock_llm_client.generate_structured.side_effect = StructuredOutputError(
            "Failed to parse JSON"
        )

        criterion = LLMCriterion(
            name="faithfulness",
            prompts_dir=prompts_dir,
            language="pt",
            llm_client=mock_llm_client,
        )

        result = criterion.evaluate(
            context="Context",
            question="Q?",
            answer="A.",
        )

        assert result.score is None
        assert result.threshold == 0.7
        assert result.error is not None
        assert "Failed to parse JSON" in result.error
        assert result.passed is False

    def test_evaluate_handles_generic_error(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test that generic errors populate error field with None score."""
        mock_llm_client.generate_structured.side_effect = Exception("LLM error")

        criterion = LLMCriterion(
            name="faithfulness",
            prompts_dir=prompts_dir,
            language="pt",
            llm_client=mock_llm_client,
        )

        result = criterion.evaluate(
            context="Context",
            question="Q?",
            answer="A.",
        )

        assert result.score is None
        assert result.threshold == 0.7
        assert result.error is not None
        assert "LLM error" in result.error
        assert result.passed is False

    def test_evaluate_clamps_scores(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test that scores outside [0, 1] are clamped."""
        mock_llm_client.generate_structured.return_value = CriterionResponse(
            score=1.5, rationale="Too high"
        )

        criterion = LLMCriterion(
            name="faithfulness",
            prompts_dir=prompts_dir,
            language="pt",
            llm_client=mock_llm_client,
        )

        result = criterion.evaluate(
            context="Context",
            question="Q?",
            answer="A.",
        )

        assert result.score == 1.0  # Clamped

    def test_evaluate_clamps_negative_scores(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test that negative scores are clamped to 0."""
        mock_llm_client.generate_structured.return_value = CriterionResponse(
            score=-0.5, rationale="Too low"
        )

        criterion = LLMCriterion(
            name="faithfulness",
            prompts_dir=prompts_dir,
            language="pt",
            llm_client=mock_llm_client,
        )

        result = criterion.evaluate(
            context="Context",
            question="Q?",
            answer="A.",
        )

        assert result.score == 0.0  # Clamped

    def test_evaluate_with_extra_params(
        self,
        mock_llm_client: Any,
        tmp_path: Path,
    ) -> None:
        """Test evaluation with criterion-specific extra parameters."""
        # Create criterion with extra params in template
        criterion_dir = tmp_path / "criteria" / "test" / "pt"
        criterion_dir.mkdir(parents=True)
        (criterion_dir / "prompt.md").write_text(
            "Context: $context\nQuestion: $question\nAnswer: $answer\nExtra: $extra_param\n"
        )
        config_file = tmp_path / "criteria" / "test" / "config.json"
        config_file.write_text(json.dumps({"threshold": 0.5}))

        mock_llm_client.generate_structured.return_value = CriterionResponse(
            score=0.6, rationale="OK"
        )

        criterion = LLMCriterion(
            name="test",
            prompts_dir=tmp_path / "criteria",
            language="pt",
            llm_client=mock_llm_client,
        )

        criterion.evaluate(
            context="Context",
            question="Q?",
            answer="A.",
            extra_param="custom_value",
        )

        # Check that prompt was built with extra param
        call_args = mock_llm_client.generate_structured.call_args
        prompt = call_args.kwargs["prompt"]
        assert "Extra: custom_value" in prompt


class _DummyHeuristicCriterion(HeuristicCriterion):
    """Concrete subclass for testing the abstract base class."""

    name = "dummy"
    threshold = 0.5

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Return score based on kwargs."""
        value: float = kwargs["value"]
        return value, f"Got value {value}"


class _FailingHeuristicCriterion(HeuristicCriterion):
    """Concrete subclass that always raises in _check."""

    name = "failing"
    threshold = 0.5

    def _check(self, **kwargs: Any) -> tuple[float, str]:
        """Always raise."""
        msg = "intentional failure"
        raise ValueError(msg)


class TestHeuristicCriterion:
    """Tests for HeuristicCriterion abstract base class."""

    def test_satisfies_judge_criterion_protocol(self) -> None:
        """Test that HeuristicCriterion subclass satisfies JudgeCriterion."""
        from arandu.shared.judge.criterion import JudgeCriterion

        criterion = _DummyHeuristicCriterion()
        assert isinstance(criterion, JudgeCriterion)

    def test_evaluate_returns_criterion_score(self) -> None:
        """Test that evaluate returns a CriterionScore."""
        criterion = _DummyHeuristicCriterion()
        result = criterion.evaluate(value=0.8)

        assert isinstance(result, CriterionScore)
        assert result.score == 0.8
        assert result.threshold == 0.5
        assert result.rationale == "Got value 0.8"
        assert result.error is None
        assert result.passed is True

    def test_evaluate_below_threshold_does_not_pass(self) -> None:
        """Test that score below threshold does not pass."""
        criterion = _DummyHeuristicCriterion()
        result = criterion.evaluate(value=0.3)

        assert result.score == 0.3
        assert result.passed is False

    def test_evaluate_wraps_exception_as_error(self) -> None:
        """Test that exceptions are caught and returned as error CriterionScore."""
        criterion = _FailingHeuristicCriterion()
        result = criterion.evaluate()

        assert result.score is None
        assert result.error is not None
        assert "intentional failure" in result.error
        assert result.passed is False

    def test_name_and_threshold_from_class_attrs(self) -> None:
        """Test that name and threshold are set from class attributes."""
        criterion = _DummyHeuristicCriterion()
        assert criterion.name == "dummy"
        assert criterion.threshold == 0.5

    def test_cannot_instantiate_abstract(self) -> None:
        """Test that HeuristicCriterion cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract method"):
            HeuristicCriterion()  # type: ignore[abstract]
