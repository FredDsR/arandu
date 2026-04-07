"""Tests for QAJudge module using shared judge pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from arandu.qa.cep.judge import QAJudge
from arandu.qa.config import CEPConfig, JudgeConfig
from arandu.qa.schemas import QAPairCEP, QAPairValidated
from arandu.shared.judge.judge import BaseJudge
from arandu.shared.judge.schemas import (
    CriterionScore,
    JudgePipelineResult,
    JudgeStepResult,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _make_pipeline_result(
    faithfulness: float = 0.9,
    bloom_calibration: float = 0.8,
    informativeness: float = 0.7,
    self_containedness: float = 0.95,
    *,
    passed: bool = True,
    thresholds: dict[str, float] | None = None,
) -> JudgePipelineResult:
    """Build a JudgePipelineResult with given criterion scores."""
    _thresholds = thresholds or {
        "faithfulness": 0.7,
        "bloom_calibration": 0.6,
        "informativeness": 0.6,
        "self_containedness": 0.6,
    }
    step_result = JudgeStepResult(
        criterion_scores={
            "faithfulness": CriterionScore(
                score=faithfulness,
                threshold=_thresholds["faithfulness"],
                rationale="Faithfulness OK",
            ),
            "bloom_calibration": CriterionScore(
                score=bloom_calibration,
                threshold=_thresholds["bloom_calibration"],
                rationale="Bloom OK",
            ),
            "informativeness": CriterionScore(
                score=informativeness,
                threshold=_thresholds["informativeness"],
                rationale="Informativeness OK",
            ),
            "self_containedness": CriterionScore(
                score=self_containedness,
                threshold=_thresholds["self_containedness"],
                rationale="Self-containedness OK",
            ),
        }
    )
    return JudgePipelineResult(
        stage_results={"cep_validation": step_result},
        passed=passed,
        rejected_at=None if passed else "cep_validation",
    )


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
        validation_threshold=0.6,
        faithfulness_weight=0.30,
        bloom_calibration_weight=0.25,
        informativeness_weight=0.25,
        self_containedness_weight=0.20,
        language="pt",
    )


@pytest.fixture
def judge_config() -> JudgeConfig:
    """Create a JudgeConfig for testing."""
    return JudgeConfig(
        language="pt",
    )


@pytest.fixture
def sample_qa_pair() -> QAPairCEP:
    """Create a sample QA pair for validation."""
    return QAPairCEP(
        question="Por que o pescador guarda o barco?",
        answer="Para evitar perda durante enchentes.",
        context="Se o rio sobe rapido, guardo o barco para evitar perda.",
        question_type="conceptual",
        confidence=0.9,
        bloom_level="analyze",
    )


class TestQAJudgeInit:
    """Tests for QAJudge initialization."""

    def test_initialization(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        judge_config: JudgeConfig,
        mocker: MockerFixture,
    ) -> None:
        """Test judge initializes with pipeline."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
            judge_config=judge_config,
        )

        assert judge.cep_config == cep_config
        assert judge.judge_config == judge_config

    def test_default_judge_config(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        mocker: MockerFixture,
    ) -> None:
        """Test judge loads default config when none provided."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
        )

        assert judge.judge_config is not None
        assert judge.judge_config.language == "pt"

    def test_is_base_judge_subclass(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        judge_config: JudgeConfig,
        mocker: MockerFixture,
    ) -> None:
        """Test that QAJudge is a BaseJudge subclass."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
            judge_config=judge_config,
        )

        assert isinstance(judge, BaseJudge)

    def test_no_validator_client_attr(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        judge_config: JudgeConfig,
        mocker: MockerFixture,
    ) -> None:
        """Test that QAJudge no longer stores validator_client directly."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
            judge_config=judge_config,
        )

        assert not hasattr(judge, "validator_client")


class TestQAJudgeValidate:
    """Tests for QAJudge.validate()."""

    def test_validate_returns_validated_pair(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        judge_config: JudgeConfig,
        sample_qa_pair: QAPairCEP,
        mocker: MockerFixture,
    ) -> None:
        """Test that validate returns a QAPairValidated with scores."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
            judge_config=judge_config,
        )

        pipeline_result = _make_pipeline_result()
        judge._pipeline = MagicMock()
        judge._pipeline.evaluate.return_value = pipeline_result

        result = judge.validate(sample_qa_pair, "context")

        assert isinstance(result, QAPairValidated)
        assert result.validation is not None
        assert result.validation.passed is True
        stage = result.validation.stage_results["cep_validation"]
        assert stage.criterion_scores["faithfulness"].score == 0.9
        assert stage.criterion_scores["bloom_calibration"].score == 0.8
        assert stage.criterion_scores["informativeness"].score == 0.7

    def test_validate_is_valid_when_passed(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        judge_config: JudgeConfig,
        sample_qa_pair: QAPairCEP,
        mocker: MockerFixture,
    ) -> None:
        """Test that is_valid reflects pipeline pass status."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
            judge_config=judge_config,
        )

        judge._pipeline = MagicMock()
        judge._pipeline.evaluate.return_value = _make_pipeline_result(passed=True)

        result = judge.validate(sample_qa_pair, "context")
        assert result.is_valid is True

    def test_validate_not_valid_when_rejected(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        judge_config: JudgeConfig,
        sample_qa_pair: QAPairCEP,
        mocker: MockerFixture,
    ) -> None:
        """Test that is_valid is False when pipeline rejects."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
            judge_config=judge_config,
        )

        judge._pipeline = MagicMock()
        judge._pipeline.evaluate.return_value = _make_pipeline_result(
            faithfulness=0.3, passed=False
        )

        result = judge.validate(sample_qa_pair, "context")
        assert result.is_valid is False

    def test_validate_handles_error(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        judge_config: JudgeConfig,
        sample_qa_pair: QAPairCEP,
        mocker: MockerFixture,
    ) -> None:
        """Test that errors result in unvalidated pair."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
            judge_config=judge_config,
        )

        judge._pipeline = MagicMock()
        judge._pipeline.evaluate.side_effect = Exception("LLM error")

        result = judge.validate(sample_qa_pair, "context")

        assert result.validation is None
        assert result.is_valid is True

    def test_validate_preserves_fields(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        judge_config: JudgeConfig,
        mocker: MockerFixture,
    ) -> None:
        """Test that all QAPairCEP fields are preserved."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        pair = QAPairCEP(
            question="Q?",
            answer="A.",
            context="Context.",
            question_type="conceptual",
            confidence=0.9,
            bloom_level="analyze",
            generation_prompt="The original prompt",
        )

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
            judge_config=judge_config,
        )

        judge._pipeline = MagicMock()
        judge._pipeline.evaluate.return_value = _make_pipeline_result()

        result = judge.validate(pair, "context")
        assert result.generation_prompt == "The original prompt"
        assert result.question == "Q?"
        assert result.answer == "A."


class TestQAJudgeBatch:
    """Tests for QAJudge.validate_batch()."""

    def test_validate_batch(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        judge_config: JudgeConfig,
        sample_qa_pair: QAPairCEP,
        mocker: MockerFixture,
    ) -> None:
        """Test batch validation of multiple QA pairs."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
            judge_config=judge_config,
        )

        judge._pipeline = MagicMock()
        judge._pipeline.evaluate.return_value = _make_pipeline_result()

        pairs = [sample_qa_pair, sample_qa_pair]
        results = judge.validate_batch(pairs, "context")

        assert len(results) == 2
        assert all(isinstance(r, QAPairValidated) for r in results)
        assert all(r.is_valid for r in results)


class TestRememberLevelPipeline:
    """Tests for remember-level pipeline (no self_containedness)."""

    def test_remember_skips_self_containedness(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        judge_config: JudgeConfig,
        mocker: MockerFixture,
    ) -> None:
        """Remember-level pairs use a pipeline without self_containedness."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        remember_pair = QAPairCEP(
            question="Quem mencionou algo no texto?",
            answer="O pescador.",
            context="O pescador mencionou a enchente.",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
        )

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
            judge_config=judge_config,
        )

        # Mock the remember pipeline (3 criteria, no self_containedness)
        remember_result = JudgePipelineResult(
            stage_results={
                "cep_validation": JudgeStepResult(
                    criterion_scores={
                        "faithfulness": CriterionScore(
                            score=0.9,
                            threshold=0.7,
                            rationale="OK",
                        ),
                        "bloom_calibration": CriterionScore(
                            score=0.8,
                            threshold=0.6,
                            rationale="OK",
                        ),
                        "informativeness": CriterionScore(
                            score=0.7,
                            threshold=0.6,
                            rationale="OK",
                        ),
                    }
                )
            },
            passed=True,
        )
        judge._remember_pipeline = MagicMock()
        judge._remember_pipeline.evaluate.return_value = remember_result

        result = judge.validate(remember_pair, "context")

        stage = result.validation.stage_results["cep_validation"]
        assert "self_containedness" not in stage.criterion_scores
        judge._remember_pipeline.evaluate.assert_called_once()

    def test_non_remember_includes_self_containedness(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        judge_config: JudgeConfig,
        sample_qa_pair: QAPairCEP,
        mocker: MockerFixture,
    ) -> None:
        """Non-remember pairs use the default pipeline with self_containedness."""
        mocker.patch("arandu.qa.cep.judge.LLMCriterionFactory")

        judge = QAJudge(
            validator_client=mock_llm_client,
            cep_config=cep_config,
            judge_config=judge_config,
        )

        judge._pipeline = MagicMock()
        judge._pipeline.evaluate.return_value = _make_pipeline_result(
            self_containedness=0.3,
        )

        result = judge.validate(sample_qa_pair, "context")

        stage = result.validation.stage_results["cep_validation"]
        assert "self_containedness" in stage.criterion_scores
        assert stage.criterion_scores["self_containedness"].score == 0.3


class TestQAPairValidatedSchema:
    """Tests for updated QAPairValidated schema."""

    def test_pair_with_judge_pipeline_result(self) -> None:
        """Test QAPairValidated with JudgePipelineResult validation."""
        result = _make_pipeline_result()
        pair = QAPairValidated(
            question="Test?",
            answer="Answer.",
            context="Context.",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
            validation=result,
            is_valid=True,
        )

        assert pair.validation is not None
        assert pair.validation.passed is True
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

    def test_pair_serialization_roundtrip(self) -> None:
        """Test QAPairValidated serializes and deserializes."""
        result = _make_pipeline_result()
        pair = QAPairValidated(
            question="Test?",
            answer="Answer.",
            context="Context.",
            question_type="factual",
            confidence=0.9,
            bloom_level="remember",
            validation=result,
            is_valid=True,
        )

        json_str = pair.model_dump_json()
        restored = QAPairValidated.model_validate_json(json_str)

        assert restored.validation is not None
        assert restored.validation.passed is True
        stage = restored.validation.stage_results["cep_validation"]
        assert stage.criterion_scores["faithfulness"].score == 0.9
