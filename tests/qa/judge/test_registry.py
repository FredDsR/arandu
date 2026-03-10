"""Tests for judge registry module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from arandu.qa.judge.criterion import FileCriterion
from arandu.qa.judge.registry import JudgeRegistry

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


class TestJudgeRegistry:
    """Tests for JudgeRegistry class."""

    def test_initialization(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test registry initialization."""
        registry = JudgeRegistry(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        assert registry.llm_client == mock_llm_client
        assert registry.language == "pt"
        assert registry.prompts_dir == prompts_dir

    def test_get_criterion_creates_new(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test getting criterion creates new FileCriterion."""
        registry = JudgeRegistry(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        criterion = registry.get_criterion("faithfulness")

        assert isinstance(criterion, FileCriterion)
        assert criterion.name == "faithfulness"
        assert criterion.language == "pt"

    def test_get_criterion_caches_result(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test getting same criterion returns cached instance."""
        registry = JudgeRegistry(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        criterion1 = registry.get_criterion("faithfulness")
        criterion2 = registry.get_criterion("faithfulness")

        assert criterion1 is criterion2  # Same object

    def test_get_criterion_missing_files(
        self,
        mock_llm_client: Any,
        tmp_path: Path,
    ) -> None:
        """Test getting criterion with missing files raises error."""
        registry = JudgeRegistry(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=tmp_path,
        )

        with pytest.raises(FileNotFoundError):
            registry.get_criterion("nonexistent")

    def test_get_criteria_cep_validation(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test getting CEP validation criterion set."""
        registry = JudgeRegistry(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        criteria = registry.get_criteria("cep_validation")

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
        registry = JudgeRegistry(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        with pytest.raises(ValueError, match="Unknown criterion set"):
            registry.get_criteria("nonexistent_set")

    def test_register_custom_criterion(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test registering custom criterion implementation."""
        registry = JudgeRegistry(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
        )

        # Create mock custom criterion
        custom_criterion = mocker.MagicMock()
        custom_criterion.name = "custom_metric"

        registry.register_custom_criterion(custom_criterion)

        # Verify it's registered
        retrieved = registry.get_criterion("custom_metric")
        assert retrieved is custom_criterion

    def test_default_prompts_dir(
        self,
        mock_llm_client: Any,
    ) -> None:
        """Test default prompts directory is set correctly."""
        registry = JudgeRegistry(
            llm_client=mock_llm_client,
            language="pt",
        )

        # Should have default path to repo prompts/judge/criteria
        assert "prompts" in str(registry.prompts_dir)
        assert "judge" in str(registry.prompts_dir)
        assert "criteria" in str(registry.prompts_dir)

    def test_temperature_and_max_tokens_passed_to_criteria(
        self,
        mock_llm_client: Any,
        prompts_dir: Path,
    ) -> None:
        """Test temperature and max_tokens are passed to created criteria."""
        registry = JudgeRegistry(
            llm_client=mock_llm_client,
            language="pt",
            prompts_dir=prompts_dir,
            temperature=0.5,
            max_tokens=4096,
        )

        criterion = registry.get_criterion("faithfulness")

        assert isinstance(criterion, FileCriterion)
        assert criterion.temperature == 0.5
        assert criterion.max_tokens == 4096
