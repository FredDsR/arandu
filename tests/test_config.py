"""Tests for configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from gtranscriber.config import (
    CEPConfig,
    EvaluationConfig,
    KGConfig,
    LLMConfig,
    QAConfig,
    TranscriberConfig,
    _get_default_temp_dir,
    get_cep_config,
    get_evaluation_config,
    get_kg_config,
    get_llm_config,
    get_qa_config,
    get_transcriber_config,
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


class TestCEPConfig:
    """Tests for CEPConfig."""

    def test_default_initialization(self) -> None:
        """Test default CEP configuration initialization."""
        config = CEPConfig()

        assert config.enable_bloom_scaffolding is True
        assert config.enable_reasoning_traces is True
        assert config.enable_validation is True
        assert config.bloom_levels == ["remember", "understand", "analyze", "evaluate"]
        assert config.max_hop_count == 3
        assert config.validator_provider == "ollama"
        assert config.validator_model_id == "llama3.1:8b"
        assert config.validator_temperature == 0.3
        assert config.validation_threshold == 0.6
        assert config.language == "pt"

    def test_valid_bloom_levels(self) -> None:
        """Test valid Bloom taxonomy levels."""
        config = CEPConfig(bloom_levels=["remember", "understand", "apply", "analyze"])
        assert config.bloom_levels == ["remember", "understand", "apply", "analyze"]

    def test_invalid_bloom_level(self) -> None:
        """Test validation error for invalid Bloom level."""
        with pytest.raises(ValidationError) as exc_info:
            CEPConfig(bloom_levels=["remember", "invalid_level"])
        assert "Invalid Bloom level" in str(exc_info.value)

    def test_valid_bloom_distribution(self) -> None:
        """Test valid Bloom distribution summing to 1.0."""
        config = CEPConfig(
            bloom_distribution={
                "remember": 0.25,
                "understand": 0.25,
                "analyze": 0.25,
                "evaluate": 0.25,
            }
        )
        assert sum(config.bloom_distribution.values()) == 1.0

    def test_invalid_bloom_distribution_sum(self) -> None:
        """Test validation error when Bloom distribution doesn't sum to 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            CEPConfig(
                bloom_distribution={
                    "remember": 0.5,
                    "understand": 0.5,
                    "analyze": 0.5,  # Sum = 1.5, invalid
                }
            )
        assert "must sum to 1.0" in str(exc_info.value)

    def test_valid_language(self) -> None:
        """Test valid language codes for CEP."""
        config_pt = CEPConfig(language="pt")
        config_en = CEPConfig(language="en")
        assert config_pt.language == "pt"
        assert config_en.language == "en"

    def test_invalid_language(self) -> None:
        """Test validation error for invalid CEP language."""
        with pytest.raises(ValidationError) as exc_info:
            CEPConfig(language="fr")
        assert "Invalid CEP language" in str(exc_info.value)

    def test_valid_scoring_weights(self) -> None:
        """Test valid scoring weights summing to 1.0."""
        config = CEPConfig(
            faithfulness_weight=0.5,
            bloom_calibration_weight=0.3,
            informativeness_weight=0.2,
        )
        total = (
            config.faithfulness_weight
            + config.bloom_calibration_weight
            + config.informativeness_weight
        )
        assert 0.99 <= total <= 1.01

    def test_invalid_scoring_weights_sum(self) -> None:
        """Test validation error when scoring weights don't sum to 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            CEPConfig(
                faithfulness_weight=0.5,
                bloom_calibration_weight=0.5,
                informativeness_weight=0.5,  # Sum = 1.5, invalid
            )
        assert "Scoring weights must sum to 1.0" in str(exc_info.value)

    def test_env_var_override(self, monkeypatch: MonkeyPatch) -> None:
        """Test CEP config loading from environment variables."""
        monkeypatch.setenv("GTRANSCRIBER_CEP_VALIDATOR_MODEL_ID", "gpt-4")
        monkeypatch.setenv("GTRANSCRIBER_CEP_ENABLE_VALIDATION", "false")

        config = CEPConfig()

        assert config.validator_model_id == "gpt-4"
        assert config.enable_validation is False

    def test_max_hop_count_boundaries(self) -> None:
        """Test max_hop_count boundary values."""
        config_min = CEPConfig(max_hop_count=1)
        config_max = CEPConfig(max_hop_count=5)
        assert config_min.max_hop_count == 1
        assert config_max.max_hop_count == 5

    def test_max_hop_count_below_min(self) -> None:
        """Test validation error when max_hop_count is below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            CEPConfig(max_hop_count=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_max_hop_count_above_max(self) -> None:
        """Test validation error when max_hop_count is above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            CEPConfig(max_hop_count=6)
        assert "less than or equal to 5" in str(exc_info.value)


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_initialization(self) -> None:
        """Test default LLM configuration initialization."""
        config = LLMConfig()

        assert config.openai_api_key is None
        assert config.base_url is None

    def test_openai_api_key_from_env(self, monkeypatch: MonkeyPatch) -> None:
        """Test OpenAI API key loaded from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")

        config = LLMConfig()

        assert config.openai_api_key == "sk-test-key-123"

    def test_base_url_from_env(self, monkeypatch: MonkeyPatch) -> None:
        """Test base URL loaded from environment."""
        monkeypatch.setenv("GTRANSCRIBER_LLM_BASE_URL", "http://custom-llm.example.com")

        config = LLMConfig()

        assert config.base_url == "http://custom-llm.example.com"


class TestQAConfigLanguage:
    """Tests for QAConfig language validation."""

    def test_valid_language_pt(self) -> None:
        """Test valid Portuguese language code."""
        config = QAConfig(language="pt")
        assert config.language == "pt"

    def test_valid_language_en(self) -> None:
        """Test valid English language code."""
        config = QAConfig(language="en")
        assert config.language == "en"

    def test_invalid_language(self) -> None:
        """Test validation error for invalid QA language."""
        with pytest.raises(ValidationError) as exc_info:
            QAConfig(language="de")
        assert "Invalid QA language" in str(exc_info.value)


class TestConfigHelperFunctions:
    """Tests for configuration helper functions."""

    def test_get_default_temp_dir(self) -> None:
        """Test _get_default_temp_dir returns gtranscriber path."""
        temp_dir = _get_default_temp_dir()
        assert "gtranscriber" in temp_dir

    def test_get_transcriber_config(self) -> None:
        """Test get_transcriber_config returns TranscriberConfig instance."""
        config = get_transcriber_config()
        assert isinstance(config, TranscriberConfig)

    def test_get_qa_config(self) -> None:
        """Test get_qa_config returns QAConfig instance."""
        config = get_qa_config()
        assert isinstance(config, QAConfig)

    def test_get_cep_config(self) -> None:
        """Test get_cep_config returns CEPConfig instance."""
        config = get_cep_config()
        assert isinstance(config, CEPConfig)

    def test_get_kg_config(self) -> None:
        """Test get_kg_config returns KGConfig instance."""
        config = get_kg_config()
        assert isinstance(config, KGConfig)

    def test_get_evaluation_config(self) -> None:
        """Test get_evaluation_config returns EvaluationConfig instance."""
        config = get_evaluation_config()
        assert isinstance(config, EvaluationConfig)

    def test_get_llm_config(self) -> None:
        """Test get_llm_config returns LLMConfig instance."""
        config = get_llm_config()
        assert isinstance(config, LLMConfig)
