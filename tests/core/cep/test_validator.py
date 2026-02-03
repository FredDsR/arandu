"""Tests for LLM-as-a-Judge Validator module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from gtranscriber.config import CEPConfig
from gtranscriber.core.cep.validator import QAValidator
from gtranscriber.schemas import QAPairCEP, QAPairValidated, ValidationScore

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    """Create a mock LLM client with required attributes."""
    client = mocker.MagicMock()
    client.provider.value = "ollama"
    client.model_id = "llama3.1:8b"
    return client


@pytest.fixture
def cep_config() -> CEPConfig:
    """Create a CEP config for testing."""
    return CEPConfig(
        enable_validation=True,
        validation_threshold=0.6,
        faithfulness_weight=0.4,
        bloom_calibration_weight=0.3,
        informativeness_weight=0.3,
        language="pt",
    )


@pytest.fixture
def sample_qa_pair() -> QAPairCEP:
    """Create a sample QA pair for validation."""
    return QAPairCEP(
        question="Por que o pescador guarda o barco?",
        answer="Para evitar perda durante enchentes.",
        context="Se o rio sobe rápido, guardo o barco para evitar perda.",
        question_type="conceptual",
        confidence=0.9,
        bloom_level="analyze",
    )


class TestQAValidator:
    """Tests for QAValidator class."""

    def test_initialization(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test validator initialization."""
        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        assert validator.validator_client == mock_llm_client
        assert validator.cep_config == cep_config

    def test_validate_returns_validated_pair(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test that validate returns a QAPairValidated with scores."""
        mock_llm_client.generate.return_value = json.dumps(
            {
                "faithfulness": 0.9,
                "bloom_calibration": 0.8,
                "informativeness": 0.7,
                "judge_rationale": "Good quality pair.",
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        assert isinstance(result, QAPairValidated)
        assert result.validation is not None
        assert result.validation.faithfulness == 0.9
        assert result.validation.bloom_calibration == 0.8
        assert result.validation.informativeness == 0.7
        assert result.validation.judge_rationale == "Good quality pair."

    def test_validate_calculates_overall_score(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test that overall score is calculated correctly."""
        mock_llm_client.generate.return_value = json.dumps(
            {
                "faithfulness": 1.0,
                "bloom_calibration": 1.0,
                "informativeness": 1.0,
                "judge_rationale": "Perfect.",
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        # Weighted average: 0.4*1.0 + 0.3*1.0 + 0.3*1.0 = 1.0
        assert result.validation.overall_score == 1.0

    def test_validate_is_valid_above_threshold(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test that is_valid is True when overall score >= threshold."""
        mock_llm_client.generate.return_value = json.dumps(
            {
                "faithfulness": 0.8,
                "bloom_calibration": 0.7,
                "informativeness": 0.6,
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        # Overall: 0.4*0.8 + 0.3*0.7 + 0.3*0.6 = 0.32 + 0.21 + 0.18 = 0.71
        # 0.71 >= 0.6 (threshold)
        assert result.is_valid is True

    def test_validate_is_valid_below_threshold(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test that is_valid is False when overall score < threshold."""
        mock_llm_client.generate.return_value = json.dumps(
            {
                "faithfulness": 0.3,
                "bloom_calibration": 0.4,
                "informativeness": 0.2,
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        # Overall: 0.4*0.3 + 0.3*0.4 + 0.3*0.2 = 0.12 + 0.12 + 0.06 = 0.30
        # 0.30 < 0.6 (threshold)
        assert result.is_valid is False

    def test_validate_handles_markdown_response(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test parsing response wrapped in markdown code block."""
        mock_llm_client.generate.return_value = """```json
{
    "faithfulness": 0.9,
    "bloom_calibration": 0.8,
    "informativeness": 0.7,
    "judge_rationale": "Good."
}
```"""

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        assert result.validation.faithfulness == 0.9

    def test_validate_handles_invalid_json(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test that invalid JSON returns pair with default scores."""
        mock_llm_client.generate.return_value = "not valid json {"

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        # Invalid JSON falls back to default scores of 0.5
        assert result.validation.faithfulness == 0.5
        assert result.validation.bloom_calibration == 0.5
        assert result.validation.informativeness == 0.5
        # 0.5 overall is below 0.6 threshold
        assert result.is_valid is False

    def test_validate_handles_llm_error(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test that LLM errors result in unvalidated pair."""
        mock_llm_client.generate.side_effect = Exception("LLM error")

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        # LLM error results in validation=None and defaults to valid
        assert result.validation is None
        assert result.is_valid is True  # Default to valid when validation fails

    def test_validate_clamps_scores(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test that scores outside [0, 1] are clamped."""
        mock_llm_client.generate.return_value = json.dumps(
            {
                "faithfulness": 1.5,
                "bloom_calibration": -0.5,
                "informativeness": 0.7,
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        assert result.validation.faithfulness == 1.0  # Clamped from 1.5
        assert result.validation.bloom_calibration == 0.0  # Clamped from -0.5

    def test_validate_batch(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test batch validation of multiple QA pairs."""
        mock_llm_client.generate.return_value = json.dumps(
            {
                "faithfulness": 0.9,
                "bloom_calibration": 0.8,
                "informativeness": 0.7,
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        pairs = [sample_qa_pair, sample_qa_pair]
        results = validator.validate_batch(pairs, "context")

        assert len(results) == 2
        assert all(isinstance(r, QAPairValidated) for r in results)
        assert all(r.is_valid for r in results)


class TestValidationScore:
    """Tests for ValidationScore schema."""

    def test_valid_score_initialization(self) -> None:
        """Test valid score initialization."""
        score = ValidationScore(
            faithfulness=0.9,
            bloom_calibration=0.8,
            informativeness=0.7,
            overall_score=0.8,
            judge_rationale="Good quality.",
        )

        assert score.faithfulness == 0.9
        assert score.bloom_calibration == 0.8
        assert score.informativeness == 0.7
        assert score.overall_score == 0.8

    def test_score_boundary_values(self) -> None:
        """Test score boundary values."""
        score = ValidationScore(
            faithfulness=0.0,
            bloom_calibration=1.0,
            informativeness=0.5,
            overall_score=0.5,
        )

        assert score.faithfulness == 0.0
        assert score.bloom_calibration == 1.0


class TestQAPairValidated:
    """Tests for QAPairValidated schema."""

    def test_valid_pair_initialization(self) -> None:
        """Test valid pair initialization."""
        validation = ValidationScore(
            faithfulness=0.9,
            bloom_calibration=0.8,
            informativeness=0.7,
            overall_score=0.8,
        )

        pair = QAPairValidated(
            question="Test?",
            answer="Answer.",
            context="Context.",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
            validation=validation,
            is_valid=True,
        )

        assert pair.validation == validation
        assert pair.is_valid is True

    def test_pair_without_validation(self) -> None:
        """Test pair can be created without validation."""
        pair = QAPairValidated(
            question="Test?",
            answer="Answer.",
            context="Context.",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
            validation=None,
            is_valid=False,
        )

        assert pair.validation is None
        assert pair.is_valid is False
