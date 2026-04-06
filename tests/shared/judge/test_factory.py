"""Tests for shared judge criterion factory module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from arandu.shared.judge.criterion import FileCriterion
from arandu.shared.judge.factory import JudgeCriterionFactory

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
    """Create temporary prompts directory with all CEP criteria."""
    base_dir = tmp_path / "criteria"

    # Create all four CEP criteria
    for criterion_name in [
        "faithfulness",
        "bloom_calibration",
        "informativeness",
        "self_containedness",
    ]:
        criterion_dir = base_dir / criterion_name / "pt"
        criterion_dir.mkdir(parents=True)
        (criterion_dir / "rubric.md").write_text(f"{criterion_name} rubric")
        (criterion_dir / "prompt.md").write_text(f"{criterion_name} prompt")

    return base_dir


class TestJudgeCriterionFactory:
    """Tests for JudgeCriterionFactory class."""

    def test_initialization(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test factory initialization."""
        factory = JudgeCriterionFactory(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        assert factory.llm_client == mock_llm_client
        assert factory.language == "pt"
        assert factory.prompts_dir == prompts_dir

    def test_get_criterion_creates_new(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test getting criterion creates new FileCriterion."""
        factory = JudgeCriterionFactory(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        criterion = factory.get_criterion("faithfulness")

        assert isinstance(criterion, FileCriterion)
        assert criterion.name == "faithfulness"
        assert criterion.language == "pt"

    def test_get_criterion_caches_result(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test getting same criterion returns cached instance."""
        factory = JudgeCriterionFactory(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        criterion1 = factory.get_criterion("faithfulness")
        criterion2 = factory.get_criterion("faithfulness")

        assert criterion1 is criterion2  # Same object

    def test_get_criterion_missing_files(
        self,
        mock_llm_client: Any,
        tmp_path: Path,
    ) -> None:
        """Test getting criterion with missing files raises error."""
        factory = JudgeCriterionFactory(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=tmp_path,
        )

        with pytest.raises(FileNotFoundError):
            factory.get_criterion("nonexistent")

    def test_get_criteria_cep_validation(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test getting CEP validation criterion set."""
        factory = JudgeCriterionFactory(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        criteria = factory.get_criteria("cep_validation")

        assert len(criteria) == 4
        criterion_names = [c.name for c in criteria]
        assert "faithfulness" in criterion_names
        assert "bloom_calibration" in criterion_names
        assert "informativeness" in criterion_names
        assert "self_containedness" in criterion_names

    def test_get_criteria_unknown_set(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test getting unknown criterion set raises error."""
        factory = JudgeCriterionFactory(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        with pytest.raises(ValueError, match="Unknown criterion set"):
            factory.get_criteria("nonexistent_set")

    def test_register_custom_criterion(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test registering custom criterion implementation."""
        factory = JudgeCriterionFactory(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        # Create mock custom criterion
        custom_criterion = mocker.MagicMock()
        custom_criterion.name = "custom_metric"

        factory.register_custom_criterion(custom_criterion)

        # Verify it's registered
        retrieved = factory.get_criterion("custom_metric")
        assert retrieved is custom_criterion

    def test_default_prompts_dir(
        self,
        mock_llm_client: Any,
    ) -> None:
        """Test default prompts directory is set correctly."""
        factory = JudgeCriterionFactory(
            llm_client=mock_llm_client,
            language="pt",
        )

        # Should have default path to repo prompts/judge/criteria
        assert "prompts" in str(factory.prompts_dir)
        assert "judge" in str(factory.prompts_dir)
        assert "criteria" in str(factory.prompts_dir)

    def test_temperature_and_max_tokens_passed_to_criteria(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test temperature and max_tokens are passed to created criteria."""
        factory = JudgeCriterionFactory(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
            temperature=0.5,
            max_tokens=4096,
        )

        criterion = factory.get_criterion("faithfulness")

        assert isinstance(criterion, FileCriterion)
        assert criterion.temperature == 0.5
        assert criterion.max_tokens == 4096
