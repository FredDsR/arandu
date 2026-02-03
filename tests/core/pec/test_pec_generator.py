"""Tests for PEC QA Generator orchestrator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from gtranscriber.config import PECConfig, QAConfig
from gtranscriber.core.pec.pec_generator import PECQAGenerator
from gtranscriber.schemas import EnrichedRecord, QARecordPEC

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    """Create a mock LLM client with required attributes."""
    client = mocker.MagicMock()
    client.provider.value = "ollama"
    client.model_id = "llama3.1:8b"
    client.generate.return_value = json.dumps(
        [
            {
                "question": "O que aconteceu?",
                "answer": "Uma resposta.",
                "bloom_level": "remember",
                "confidence": 0.9,
            }
        ]
    )
    return client


@pytest.fixture
def mock_validator_client(mocker: MockerFixture) -> Any:
    """Create a mock validator LLM client with required attributes."""
    client = mocker.MagicMock()
    client.provider.value = "ollama"
    client.model_id = "llama3.1:8b"
    client.generate.return_value = json.dumps(
        {
            "faithfulness": 0.9,
            "bloom_calibration": 0.8,
            "informativeness": 0.7,
        }
    )
    return client


@pytest.fixture
def qa_config() -> QAConfig:
    """Create a QA config for testing."""
    return QAConfig(
        questions_per_document=8,
        temperature=0.7,
        max_tokens=2048,
    )


@pytest.fixture
def pec_config() -> PECConfig:
    """Create a PEC config for testing."""
    return PECConfig(
        enable_bloom_scaffolding=True,
        enable_reasoning_traces=True,
        enable_validation=False,
        bloom_levels=["remember", "understand", "analyze", "evaluate"],
        bloom_distribution={
            "remember": 0.25,
            "understand": 0.25,
            "analyze": 0.25,
            "evaluate": 0.25,
        },
        language="pt",
    )


@pytest.fixture
def sample_transcription() -> EnrichedRecord:
    """Create a sample transcription for testing."""
    return EnrichedRecord(
        gdrive_id="test123",
        name="test.mp3",
        mimeType="audio/mpeg",
        parents=["folder"],
        webContentLink="https://drive.google.com/test",
        size_bytes=1024000,
        duration_milliseconds=60000,
        transcription_text="Este é um texto de teste para geração de QA. " * 20,
        detected_language="pt",
        language_probability=0.95,
        model_id="whisper-large-v3",
        compute_device="cpu",
        processing_duration_sec=30.5,
        transcription_status="completed",
    )


class TestPECQAGenerator:
    """Tests for PECQAGenerator class."""

    def test_initialization(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        pec_config: PECConfig,
    ) -> None:
        """Test generator initialization."""
        generator = PECQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            pec_config=pec_config,
        )

        assert generator.llm_client == mock_llm_client
        assert generator.qa_config == qa_config
        assert generator.pec_config == pec_config

    def test_initialization_with_validator(
        self,
        mock_llm_client: Any,
        mock_validator_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test generator initialization with validator client."""
        pec_config = PECConfig(enable_validation=True)

        generator = PECQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            pec_config=pec_config,
            validator_client=mock_validator_client,
        )

        assert generator.validator_client == mock_validator_client

    def test_generate_qa_pairs_returns_record(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        pec_config: PECConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test that generate_qa_pairs returns a QARecordPEC."""
        generator = PECQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            pec_config=pec_config,
        )

        result = generator.generate_qa_pairs(sample_transcription)

        assert isinstance(result, QARecordPEC)
        assert result.source_gdrive_id == "test123"
        assert result.source_filename == "test.mp3"
        assert result.model_id == qa_config.model_id
        assert result.provider == qa_config.provider

    def test_generate_qa_pairs_includes_bloom_distribution(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        pec_config: PECConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test that result includes Bloom level distribution."""
        generator = PECQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            pec_config=pec_config,
        )

        result = generator.generate_qa_pairs(sample_transcription)

        assert result.bloom_distribution is not None
        # Should have distribution for all configured levels
        for level in pec_config.bloom_levels:
            assert level in result.bloom_distribution

    def test_generate_qa_pairs_with_validation(
        self,
        mock_llm_client: Any,
        mock_validator_client: Any,
        qa_config: QAConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test generation with validation enabled."""
        pec_config = PECConfig(
            enable_validation=True,
            validation_threshold=0.6,
        )

        generator = PECQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            pec_config=pec_config,
            validator_client=mock_validator_client,
        )

        result = generator.generate_qa_pairs(sample_transcription)

        assert result.validation_summary is not None
        # Validator should have been called
        assert mock_validator_client.generate.called

    def test_generate_qa_pairs_without_validation(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        pec_config: PECConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test generation without validation."""
        generator = PECQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            pec_config=pec_config,
        )

        result = generator.generate_qa_pairs(sample_transcription)

        assert result.validation_summary is None

    def test_generate_handles_empty_transcription(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        pec_config: PECConfig,
    ) -> None:
        """Test handling of empty transcription text."""
        empty_transcription = EnrichedRecord(
            gdrive_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder"],
            webContentLink="https://drive.google.com/test",
            size_bytes=1024,
            duration_milliseconds=1000,
            transcription_text="",
            detected_language="pt",
            language_probability=0.95,
            model_id="whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=1.0,
            transcription_status="completed",
        )

        generator = PECQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            pec_config=pec_config,
        )

        # Empty transcription should raise ValueError
        with pytest.raises(ValueError, match="Transcription too short"):
            generator.generate_qa_pairs(empty_transcription)

    def test_generate_handles_short_transcription(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        pec_config: PECConfig,
    ) -> None:
        """Test handling of very short transcription."""
        short_transcription = EnrichedRecord(
            gdrive_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder"],
            webContentLink="https://drive.google.com/test",
            size_bytes=1024,
            duration_milliseconds=1000,
            transcription_text="Hello.",
            detected_language="pt",
            language_probability=0.95,
            model_id="whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=1.0,
            transcription_status="completed",
        )

        generator = PECQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            pec_config=pec_config,
        )

        # Short transcription below MIN_CONTEXT_LENGTH should raise ValueError
        with pytest.raises(ValueError, match="Transcription too short"):
            generator.generate_qa_pairs(short_transcription)

    def test_to_jsonl_format(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        pec_config: PECConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test JSONL export format."""
        generator = PECQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            pec_config=pec_config,
        )

        result = generator.generate_qa_pairs(sample_transcription)
        jsonl = result.to_jsonl()

        # Should have one line per QA pair
        lines = [line for line in jsonl.strip().split("\n") if line]
        assert len(lines) == len(result.qa_pairs)

        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert "question" in data
            assert "answer" in data
            assert "context" in data


class TestQARecordPEC:
    """Tests for QARecordPEC schema."""

    def test_bloom_distribution_validation(self) -> None:
        """Test Bloom distribution validation."""
        from gtranscriber.schemas import QAPairPEC

        pairs = [
            QAPairPEC(
                question="Q1?",
                answer="A1",
                context="C1",
                question_type="factual",
                confidence=0.9,
                bloom_level="remember",
            ),
            QAPairPEC(
                question="Q2?",
                answer="A2",
                context="C2",
                question_type="conceptual",
                confidence=0.9,
                bloom_level="analyze",
            ),
        ]

        record = QARecordPEC(
            source_gdrive_id="test123",
            source_filename="test.mp3",
            transcription_text="Test text.",
            qa_pairs=pairs,
            model_id="llama3.1:8b",
            provider="ollama",
            total_pairs=2,
            bloom_distribution={"remember": 1, "analyze": 1},
        )

        assert record.bloom_distribution["remember"] == 1
        assert record.bloom_distribution["analyze"] == 1

    def test_validation_summary_optional(self) -> None:
        """Test that validation_summary is optional."""
        from gtranscriber.schemas import QAPairPEC

        pairs = [
            QAPairPEC(
                question="Q?",
                answer="A",
                context="C",
                question_type="factual",
                confidence=0.9,
                bloom_level="remember",
            ),
        ]

        record = QARecordPEC(
            source_gdrive_id="test123",
            source_filename="test.mp3",
            transcription_text="Test text.",
            qa_pairs=pairs,
            model_id="llama3.1:8b",
            provider="ollama",
            total_pairs=1,
            bloom_distribution={"remember": 1},
            validation_summary=None,
        )

        assert record.validation_summary is None

    def test_to_jsonl_with_bloom_level(self) -> None:
        """Test JSONL export includes Bloom level."""
        from gtranscriber.schemas import QAPairPEC

        pairs = [
            QAPairPEC(
                question="Por que?",
                answer="Porque sim.",
                context="Contexto de teste.",
                question_type="conceptual",
                confidence=0.9,
                bloom_level="analyze",
                reasoning_trace="A → B → C",
            ),
        ]

        record = QARecordPEC(
            source_gdrive_id="test123",
            source_filename="test.mp3",
            transcription_text="Test text.",
            qa_pairs=pairs,
            model_id="llama3.1:8b",
            provider="ollama",
            total_pairs=1,
            bloom_distribution={"analyze": 1},
        )

        jsonl = record.to_jsonl()
        data = json.loads(jsonl.strip())

        assert data["bloom_level"] == "analyze"
        assert data["reasoning_trace"] == "A → B → C"
