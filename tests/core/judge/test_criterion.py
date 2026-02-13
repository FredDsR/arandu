"""Tests for judge criterion module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from gtranscriber.core.judge.criterion import FileCriterion
from gtranscriber.schemas import CriterionScore
from gtranscriber.utils.text import GenerateResult

if TYPE_CHECKING:
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

    # Create rubric file
    rubric_file = faithfulness_dir / "rubric.md"
    rubric_file.write_text("Rubric content here")

    # Create prompt template file
    prompt_file = faithfulness_dir / "prompt.md"
    prompt_file.write_text(
        "Context: $context\n"
        "Question: $question\n"
        "Answer: $answer\n"
        "Rubric: $rubric\n"
    )

    return base_dir


class TestFileCriterion:
    """Tests for FileCriterion class."""

    def test_initialization(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test criterion initialization loads files."""
        criterion = FileCriterion(
            name="faithfulness",
            prompts_dir=prompts_dir,
            language="pt",
            llm_client=mock_llm_client,
        )

        assert criterion.name == "faithfulness"
        assert criterion.language == "pt"
        assert criterion.llm_client == mock_llm_client
        assert "Rubric content here" in criterion.rubric
        assert "$context" in criterion.prompt_template

    def test_initialization_missing_rubric(
        self,
        mock_llm_client: Any,
        tmp_path: Path,
    ) -> None:
        """Test that FileNotFoundError is raised for missing rubric."""
        criterion_dir = tmp_path / "criteria" / "test" / "pt"
        criterion_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Rubric file not found"):
            FileCriterion(
                name="test",
                prompts_dir=tmp_path / "criteria",
                language="pt",
                llm_client=mock_llm_client,
            )

    def test_initialization_missing_prompt(
        self,
        mock_llm_client: Any,
        tmp_path: Path,
    ) -> None:
        """Test that FileNotFoundError is raised for missing prompt."""
        criterion_dir = tmp_path / "criteria" / "test" / "pt"
        criterion_dir.mkdir(parents=True)
        (criterion_dir / "rubric.md").write_text("Rubric")

        with pytest.raises(FileNotFoundError, match="Prompt file not found"):
            FileCriterion(
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
        mock_llm_client.generate.return_value = GenerateResult(
            content=json.dumps({"score": 0.8, "rationale": "Good quality"})
        )

        criterion = FileCriterion(
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
        assert result.criterion_name == "faithfulness"
        assert result.score == 0.8
        assert result.rationale == "Good quality"

        # Verify LLM was called with correct params
        mock_llm_client.generate.assert_called_once()
        call_kwargs = mock_llm_client.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 1024

    def test_evaluate_with_thinking(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test evaluation stores thinking trace."""
        mock_llm_client.generate.return_value = GenerateResult(
            content=json.dumps({"score": 0.7, "rationale": "Decent"}),
            thinking="Internal reasoning process",
        )

        criterion = FileCriterion(
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

        assert result.thinking == "Internal reasoning process"

    def test_evaluate_handles_markdown_response(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test parsing response wrapped in markdown code block."""
        mock_llm_client.generate.return_value = GenerateResult(
            content='```json\n{"score": 0.9, "rationale": "Excellent"}\n```'
        )

        criterion = FileCriterion(
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

        assert result.score == 0.9
        assert result.rationale == "Excellent"

    def test_evaluate_handles_invalid_json(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test that invalid JSON returns default score."""
        mock_llm_client.generate.return_value = GenerateResult(content="not valid json")

        criterion = FileCriterion(
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

        assert result.score == 0.5  # Default
        assert "Failed to parse" in result.rationale

    def test_evaluate_handles_llm_error(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test that LLM errors return default score."""
        mock_llm_client.generate.side_effect = Exception("LLM error")

        criterion = FileCriterion(
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

        assert result.score == 0.5  # Default
        assert "Evaluation failed" in result.rationale

    def test_evaluate_clamps_scores(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test that scores outside [0, 1] are clamped."""
        mock_llm_client.generate.return_value = GenerateResult(
            content=json.dumps({"score": 1.5, "rationale": "Too high"})
        )

        criterion = FileCriterion(
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

    def test_evaluate_with_extra_params(
        self,
        mock_llm_client: Any,
        tmp_path: Path,
    ) -> None:
        """Test evaluation with criterion-specific extra parameters."""
        # Create criterion with extra params in template
        criterion_dir = tmp_path / "criteria" / "test" / "pt"
        criterion_dir.mkdir(parents=True)
        (criterion_dir / "rubric.md").write_text("Rubric")
        (criterion_dir / "prompt.md").write_text(
            "Context: $context\n"
            "Question: $question\n"
            "Answer: $answer\n"
            "Extra: $extra_param\n"
        )

        mock_llm_client.generate.return_value = GenerateResult(
            content=json.dumps({"score": 0.6, "rationale": "OK"})
        )

        criterion = FileCriterion(
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
        call_args = mock_llm_client.generate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "Extra: custom_value" in prompt

    def test_validate_score_with_invalid_types(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test _validate_score handles various invalid input types."""
        criterion = FileCriterion(
            name="faithfulness",
            prompts_dir=prompts_dir,
            language="pt",
            llm_client=mock_llm_client,
        )

        assert criterion._validate_score(None) == 0.5
        assert criterion._validate_score("not_a_number") == 0.5
        assert criterion._validate_score([1, 2, 3]) == 0.5

        # Valid types
        assert criterion._validate_score(0.7) == 0.7
        assert criterion._validate_score(1) == 1.0
        assert criterion._validate_score(0) == 0.0
