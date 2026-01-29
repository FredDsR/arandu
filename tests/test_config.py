"""Tests for configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from gtranscriber.config import (
    EvaluationConfig,
    KGConfig,
    QAConfig,
    TranscriberConfig,
)


class TestTranscriberConfig:
    """Tests for TranscriberConfig."""

    def test_default_initialization(self, monkeypatch: MonkeyPatch) -> None:
        """Test default configuration initialization."""
        # Override with the default values to test the actual config defaults
        # (not influenced by .env file)
        monkeypatch.setenv("GTRANSCRIBER_MODEL_ID", "openai/whisper-large-v3")
        monkeypatch.setenv("GTRANSCRIBER_WORKERS", "1")

        config = TranscriberConfig()

        assert config.model_id == "openai/whisper-large-v3"
        assert config.language is None
        assert config.return_timestamps is True
        assert config.chunk_length_s == 30
        assert config.stride_length_s == 5
        assert config.force_cpu is False
        assert config.quantize is False
        assert config.quantize_bits == 8
        assert config.workers == 1
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_env_var_override(self, monkeypatch: MonkeyPatch) -> None:
        """Test configuration loading from environment variables."""
        monkeypatch.setenv("GTRANSCRIBER_MODEL_ID", "custom/model")
        monkeypatch.setenv("GTRANSCRIBER_LANGUAGE", "en")
        monkeypatch.setenv("GTRANSCRIBER_FORCE_CPU", "true")
        monkeypatch.setenv("GTRANSCRIBER_WORKERS", "4")

        config = TranscriberConfig()

        assert config.model_id == "custom/model"
        assert config.language == "en"
        assert config.force_cpu is True
        assert config.workers == 4

    def test_credentials_file_property(self) -> None:
        """Test backward compatibility property for credentials_file."""
        config = TranscriberConfig(credentials="custom_creds.json")
        assert config.credentials_file == "custom_creds.json"

    def test_token_file_property(self) -> None:
        """Test backward compatibility property for token_file."""
        config = TranscriberConfig(token="custom_token.json")
        assert config.token_file == "custom_token.json"

    def test_temp_dir_default(self) -> None:
        """Test default temp directory creation."""
        config = TranscriberConfig()
        assert "gtranscriber" in config.temp_dir

    def test_path_fields(self) -> None:
        """Test path configuration fields."""
        config = TranscriberConfig(
            input_dir="./test_input",
            results_dir="./test_results",
            credentials_dir="./test_creds",
            hf_cache_dir="./test_cache",
        )

        assert config.input_dir == "./test_input"
        assert config.results_dir == "./test_results"
        assert config.credentials_dir == "./test_creds"
        assert config.hf_cache_dir == "./test_cache"


class TestQAConfig:
    """Tests for QAConfig."""

    def test_default_initialization(self) -> None:
        """Test default QA configuration initialization."""
        config = QAConfig()

        assert config.provider == "ollama"
        assert config.model_id == "llama3.1:8b"
        assert config.ollama_url == "http://localhost:11434/v1"
        assert config.base_url is None
        assert config.questions_per_document == 10
        assert config.strategies == ["factual", "conceptual"]
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.workers == 2

    def test_questions_per_document_boundary_min(self) -> None:
        """Test minimum boundary for questions_per_document."""
        config = QAConfig(questions_per_document=1)
        assert config.questions_per_document == 1

    def test_questions_per_document_boundary_max(self) -> None:
        """Test maximum boundary for questions_per_document."""
        config = QAConfig(questions_per_document=50)
        assert config.questions_per_document == 50

    def test_questions_per_document_below_min(self) -> None:
        """Test validation error when questions_per_document is below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            QAConfig(questions_per_document=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_questions_per_document_above_max(self) -> None:
        """Test validation error when questions_per_document is above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            QAConfig(questions_per_document=51)
        assert "less than or equal to 50" in str(exc_info.value)

    def test_temperature_boundary_min(self) -> None:
        """Test minimum boundary for temperature."""
        config = QAConfig(temperature=0.0)
        assert config.temperature == 0.0

    def test_temperature_boundary_max(self) -> None:
        """Test maximum boundary for temperature."""
        config = QAConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_temperature_below_min(self) -> None:
        """Test validation error when temperature is below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            QAConfig(temperature=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_temperature_above_max(self) -> None:
        """Test validation error when temperature is above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            QAConfig(temperature=2.1)
        assert "less than or equal to 2" in str(exc_info.value)

    def test_valid_strategies(self) -> None:
        """Test valid QA generation strategies."""
        config = QAConfig(strategies=["factual", "conceptual", "temporal", "entity"])
        assert config.strategies == ["factual", "conceptual", "temporal", "entity"]

    def test_invalid_strategy(self) -> None:
        """Test validation error for invalid strategy."""
        with pytest.raises(ValidationError) as exc_info:
            QAConfig(strategies=["factual", "invalid_strategy"])
        assert "Invalid QA strategy" in str(exc_info.value)

    def test_env_var_override(self, monkeypatch: MonkeyPatch) -> None:
        """Test QA config loading from environment variables."""
        monkeypatch.setenv("GTRANSCRIBER_QA_PROVIDER", "openai")
        monkeypatch.setenv("GTRANSCRIBER_QA_MODEL_ID", "gpt-4")
        monkeypatch.setenv("GTRANSCRIBER_QA_TEMPERATURE", "0.9")

        config = QAConfig()

        assert config.provider == "openai"
        assert config.model_id == "gpt-4"
        assert config.temperature == 0.9

    def test_output_dir_path_type(self) -> None:
        """Test output_dir is a Path object."""
        config = QAConfig(output_dir="custom_qa_output")
        assert isinstance(config.output_dir, Path)
        assert str(config.output_dir) == "custom_qa_output"


class TestKGConfig:
    """Tests for KGConfig."""

    def test_default_initialization(self) -> None:
        """Test default KG configuration initialization."""
        config = KGConfig()

        assert config.provider == "ollama"
        assert config.model_id == "llama3.1:8b"
        assert config.ollama_url == "http://localhost:11434/v1"
        assert config.base_url is None
        assert config.workers == 2

    def test_env_var_override(self, monkeypatch: MonkeyPatch) -> None:
        """Test KG config loading from environment variables."""
        monkeypatch.setenv("GTRANSCRIBER_KG_PROVIDER", "openai")
        monkeypatch.setenv("GTRANSCRIBER_KG_MODEL_ID", "gpt-3.5-turbo")

        config = KGConfig()

        assert config.provider == "openai"
        assert config.model_id == "gpt-3.5-turbo"


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_default_initialization(self) -> None:
        """Test default Evaluation configuration initialization."""
        config = EvaluationConfig()

        assert config.metrics == ["qa", "entity", "relation", "semantic"]
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert isinstance(config.output_dir, Path)
        assert str(config.output_dir) == "evaluation"
        assert isinstance(config.qa_dir, Path)
        assert isinstance(config.kg_dir, Path)

    def test_env_var_override(self, monkeypatch: MonkeyPatch) -> None:
        """Test Evaluation config loading from environment variables."""
        monkeypatch.setenv("GTRANSCRIBER_EVAL_EMBEDDING_MODEL", "custom-model")

        config = EvaluationConfig()

        assert config.embedding_model == "custom-model"

    def test_valid_metrics(self) -> None:
        """Test valid evaluation metrics."""
        config = EvaluationConfig(metrics=["qa", "entity"])
        assert config.metrics == ["qa", "entity"]

    def test_invalid_metric(self) -> None:
        """Test validation error for invalid metric."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationConfig(metrics=["qa", "invalid_metric"])
        assert "Invalid evaluation metric" in str(exc_info.value)


class TestConfigEnvPrefix:
    """Test environment variable prefix handling."""

    def test_transcriber_prefix(self, monkeypatch: MonkeyPatch) -> None:
        """Test GTRANSCRIBER_ prefix for TranscriberConfig."""
        monkeypatch.setenv("GTRANSCRIBER_MODEL_ID", "test-model")
        config = TranscriberConfig()
        assert config.model_id == "test-model"

    def test_qa_prefix(self, monkeypatch: MonkeyPatch) -> None:
        """Test GTRANSCRIBER_QA_ prefix for QAConfig."""
        monkeypatch.setenv("GTRANSCRIBER_QA_MODEL_ID", "test-qa-model")
        config = QAConfig()
        assert config.model_id == "test-qa-model"

    def test_kg_prefix(self, monkeypatch: MonkeyPatch) -> None:
        """Test GTRANSCRIBER_KG_ prefix for KGConfig."""
        monkeypatch.setenv("GTRANSCRIBER_KG_MODEL_ID", "test-kg-model")
        config = KGConfig()
        assert config.model_id == "test-kg-model"

    def test_eval_prefix(self, monkeypatch: MonkeyPatch) -> None:
        """Test GTRANSCRIBER_EVAL_ prefix for EvaluationConfig."""
        monkeypatch.setenv("GTRANSCRIBER_EVAL_EMBEDDING_MODEL", "test-eval-model")
        config = EvaluationConfig()
        assert config.embedding_model == "test-eval-model"

    def test_case_insensitive(self, monkeypatch: MonkeyPatch) -> None:
        """Test case insensitivity of environment variables."""
        monkeypatch.setenv("gtranscriber_model_id", "lowercase-model")
        config = TranscriberConfig()
        assert config.model_id == "lowercase-model"
