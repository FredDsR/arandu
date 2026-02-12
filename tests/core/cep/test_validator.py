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
        faithfulness_weight=0.30,
        bloom_calibration_weight=0.25,
        informativeness_weight=0.25,
        self_containedness_weight=0.20,
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
                "self_containedness": 1.0,
                "judge_rationale": "Perfect.",
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        # Weighted average: 0.30*1.0 + 0.25*1.0 + 0.25*1.0 + 0.20*1.0 = 1.0
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
                "self_containedness": 0.9,
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        # Overall: 0.30*0.8 + 0.25*0.7 + 0.25*0.6 + 0.20*0.9
        # = 0.24 + 0.175 + 0.15 + 0.18 = 0.745
        # 0.745 >= 0.6 (threshold)
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
                "self_containedness": 0.1,
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        # Overall: 0.30*0.3 + 0.25*0.4 + 0.25*0.2 + 0.20*0.1
        # = 0.09 + 0.10 + 0.05 + 0.02 = 0.26
        # 0.26 < 0.6 (threshold)
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

        # Invalid JSON falls back to default scores
        assert result.validation.faithfulness == 0.5
        assert result.validation.bloom_calibration == 0.5
        assert result.validation.informativeness == 0.5
        assert result.validation.self_containedness == 1.0
        # Overall: 0.30*0.5 + 0.25*0.5 + 0.25*0.5 + 0.20*1.0
        # = 0.15 + 0.125 + 0.125 + 0.20 = 0.60
        # 0.60 >= 0.6 (threshold) -- edge case, passes
        assert result.is_valid is True

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

    def test_validate_preserves_generation_prompt_success(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test that generation_prompt is preserved through validate() success path."""
        pair = QAPairCEP(
            question="Q?",
            answer="A.",
            context="Context.",
            question_type="conceptual",
            confidence=0.9,
            bloom_level="analyze",
            generation_prompt="The original prompt",
        )

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

        result = validator.validate(pair, "context")

        assert result.generation_prompt == "The original prompt"

    def test_validate_preserves_generation_prompt_error(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test that generation_prompt is preserved through validate() error path."""
        pair = QAPairCEP(
            question="Q?",
            answer="A.",
            context="Context.",
            question_type="conceptual",
            confidence=0.9,
            bloom_level="analyze",
            generation_prompt="The original prompt",
        )

        mock_llm_client.generate.side_effect = Exception("LLM error")

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(pair, "context")

        assert result.generation_prompt == "The original prompt"

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

    def test_validate_passes_response_format(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test that validate passes response_format to LLM client."""
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

        validator.validate(sample_qa_pair, "context")

        call_kwargs = mock_llm_client.generate.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}


class TestValidationScore:
    """Tests for ValidationScore schema."""

    def test_valid_score_initialization(self) -> None:
        """Test valid score initialization."""
        score = ValidationScore(
            faithfulness=0.9,
            bloom_calibration=0.8,
            informativeness=0.7,
            self_containedness=0.95,
            overall_score=0.8,
            judge_rationale="Good quality.",
        )

        assert score.faithfulness == 0.9
        assert score.bloom_calibration == 0.8
        assert score.informativeness == 0.7
        assert score.self_containedness == 0.95
        assert score.overall_score == 0.8

    def test_self_containedness_defaults_to_one(self) -> None:
        """Test that self_containedness defaults to 1.0 for backward compat."""
        score = ValidationScore(
            faithfulness=0.9,
            bloom_calibration=0.8,
            informativeness=0.7,
            overall_score=0.8,
        )

        assert score.self_containedness == 1.0

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


class TestQAValidatorEdgeCases:
    """Additional edge case tests for QAValidator."""

    def test_load_prompts_file_not_found(
        self,
        mock_llm_client: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test that FileNotFoundError is raised when prompt file doesn't exist."""
        cep_config = CEPConfig(
            enable_validation=True,
            language="pt",
        )

        # Mock the file existence check to return False
        mocker.patch("pathlib.Path.exists", return_value=False)

        with pytest.raises(FileNotFoundError, match="Validation data file not found"):
            QAValidator(
                validator_client=mock_llm_client,
                cep_config=cep_config,
            )

    def test_load_prompts_template_not_found(
        self,
        mock_llm_client: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test that FileNotFoundError is raised when template file doesn't exist."""
        cep_config = CEPConfig(
            enable_validation=True,
            language="pt",
        )

        mocker.patch("pathlib.Path.exists", side_effect=[True, False])

        with pytest.raises(FileNotFoundError, match="Validation template not found"):
            QAValidator(
                validator_client=mock_llm_client,
                cep_config=cep_config,
            )

    def test_validate_score_with_invalid_types(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test _validate_score handles various invalid input types."""
        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        # Test various invalid types
        assert validator._validate_score(None) == 0.5
        assert validator._validate_score("not_a_number") == 0.5
        assert validator._validate_score([1, 2, 3]) == 0.5
        assert validator._validate_score({"key": "value"}) == 0.5

        # Test valid types
        assert validator._validate_score(0.7) == 0.7
        assert validator._validate_score(1) == 1.0
        assert validator._validate_score(0) == 0.0


class TestSelfContainedness:
    """Tests for self_containedness criterion in validation."""

    def test_validate_parses_self_containedness(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test that self_containedness is parsed from LLM response."""
        mock_llm_client.generate.return_value = json.dumps(
            {
                "faithfulness": 0.9,
                "bloom_calibration": 0.8,
                "informativeness": 0.7,
                "self_containedness": 0.6,
                "judge_rationale": "Partially self-contained.",
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        assert result.validation.self_containedness == 0.6

    def test_validate_defaults_self_containedness_when_missing(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test that self_containedness defaults to 1.0 when not in response."""
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

        result = validator.validate(sample_qa_pair, "context")

        assert result.validation.self_containedness == 1.0

    def test_validate_overrides_self_containedness_for_remember(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test that self_containedness is forced to 1.0 for remember level."""
        remember_pair = QAPairCEP(
            question="Quem mencionou algo no texto?",
            answer="O pescador.",
            context="O pescador mencionou a enchente.",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
        )

        # LLM judge gives low self_containedness (incorrect for remember)
        mock_llm_client.generate.return_value = json.dumps(
            {
                "faithfulness": 0.9,
                "bloom_calibration": 0.8,
                "informativeness": 0.7,
                "self_containedness": 0.2,
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(remember_pair, "context")

        # Safety net forces 1.0 for remember level
        assert result.validation.self_containedness == 1.0

    def test_overall_score_includes_self_containedness(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair: QAPairCEP,
    ) -> None:
        """Test that overall score calculation includes self_containedness weight."""
        mock_llm_client.generate.return_value = json.dumps(
            {
                "faithfulness": 1.0,
                "bloom_calibration": 1.0,
                "informativeness": 1.0,
                "self_containedness": 0.0,
            }
        )

        validator = QAValidator(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = validator.validate(sample_qa_pair, "context")

        # Overall: 0.30*1.0 + 0.25*1.0 + 0.25*1.0 + 0.20*0.0 = 0.80
        assert abs(result.validation.overall_score - 0.80) < 0.01
