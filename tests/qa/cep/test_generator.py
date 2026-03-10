"""Tests for CEP QA Generator orchestrator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from arandu.qa.cep.generator import CEPQAGenerator
from arandu.qa.config import CEPConfig, QAConfig
from arandu.qa.schemas import QARecordCEP
from arandu.shared.schemas import EnrichedRecord
from arandu.utils.text import GenerateResult

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    """Create a mock LLM client with required attributes."""
    client = mocker.MagicMock()
    client.provider.value = "ollama"
    client.model_id = "qwen3:14b"
    client.generate.return_value = GenerateResult(
        content=json.dumps(
            {
                "question": "O que aconteceu?",
                "answer": "Uma resposta.",
                "confidence": 0.9,
            }
        ),
    )
    return client


@pytest.fixture
def mock_validator_client(mocker: MockerFixture) -> Any:
    """Create a mock validator LLM client with required attributes."""
    client = mocker.MagicMock()
    client.provider.value = "ollama"
    client.model_id = "qwen3:14b"
    client.generate.return_value = GenerateResult(
        content=json.dumps(
            {
                "faithfulness": 0.9,
                "bloom_calibration": 0.8,
                "informativeness": 0.7,
                "self_containedness": 0.9,
            }
        ),
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
def cep_config() -> CEPConfig:
    """Create a CEP config for testing."""
    return CEPConfig(
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
        file_id="test123",
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


class TestCEPQAGenerator:
    """Tests for CEPQAGenerator class."""

    def test_initialization(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test generator initialization."""
        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        assert generator.llm_client == mock_llm_client
        assert generator.qa_config == qa_config
        assert generator.cep_config == cep_config

    def test_initialization_with_validator(
        self,
        mock_llm_client: Any,
        mock_validator_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test generator initialization with validator client."""
        cep_config = CEPConfig(enable_validation=True)

        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
            validator_client=mock_validator_client,
        )

        assert generator.validator_client == mock_validator_client

    def test_generate_qa_pairs_returns_record(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test that generate_qa_pairs returns a QARecordCEP."""
        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        result = generator.generate_qa_pairs(sample_transcription)

        assert isinstance(result, QARecordCEP)
        assert result.source_file_id == "test123"
        assert result.source_filename == "test.mp3"
        assert result.model_id == qa_config.model_id
        assert result.provider == qa_config.provider

    def test_generate_qa_pairs_includes_bloom_distribution(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test that result includes Bloom level distribution."""
        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        result = generator.generate_qa_pairs(sample_transcription)

        assert result.bloom_distribution is not None
        # Should have distribution for all configured levels
        for level in cep_config.bloom_levels:
            assert level in result.bloom_distribution

    def test_generate_qa_pairs_with_validation(
        self,
        mock_llm_client: Any,
        mock_validator_client: Any,
        qa_config: QAConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test generation with validation enabled."""
        cep_config = CEPConfig(
            enable_validation=True,
            validation_threshold=0.6,
        )

        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
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
        cep_config: CEPConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test generation without validation."""
        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        result = generator.generate_qa_pairs(sample_transcription)

        assert result.validation_summary is None

    def test_generate_handles_empty_transcription(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test handling of empty transcription text."""
        empty_transcription = EnrichedRecord(
            file_id="test123",
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

        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        # Empty transcription should raise ValueError
        with pytest.raises(ValueError, match="Transcription too short"):
            generator.generate_qa_pairs(empty_transcription)

    def test_generate_handles_short_transcription(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test handling of very short transcription."""
        short_transcription = EnrichedRecord(
            file_id="test123",
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

        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        # Short transcription below MIN_CONTEXT_LENGTH should raise ValueError
        with pytest.raises(ValueError, match="Transcription too short"):
            generator.generate_qa_pairs(short_transcription)

    def test_chunk_text_short_text(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test chunking of text shorter than MAX_CONTEXT_LENGTH."""
        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        short_text = "This is a short text. It should not be chunked."
        chunks = generator._chunk_text(short_text)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunk_text_long_text_multiple_chunks(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test chunking of long text into multiple chunks."""
        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        # Create text longer than MAX_CONTEXT_LENGTH (4000 chars)
        # by repeating a sentence many times
        sentence = "This is a test sentence that will be repeated many times. "
        long_text = sentence * 100  # ~5800 chars

        chunks = generator._chunk_text(long_text)

        # Should be split into multiple chunks
        assert len(chunks) > 1
        # Each chunk should be within limit
        for chunk in chunks:
            assert len(chunk) <= 4000
        # When joined, should contain all content (roughly)
        total_length = sum(len(chunk) for chunk in chunks)
        assert total_length > 0

    def test_chunk_text_very_long_sentence(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test chunking with a single sentence longer than MAX_CONTEXT_LENGTH."""
        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        # Create a very long sentence without sentence boundaries
        very_long_sentence = "a" * 5000  # 5000 chars, no sentence boundaries

        chunks = generator._chunk_text(very_long_sentence)

        # Should be truncated to MAX_CONTEXT_LENGTH
        assert len(chunks) == 1
        assert len(chunks[0]) == 4000

    def test_chunk_text_with_sentence_boundaries(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that chunking respects sentence boundaries."""
        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        # Create text with clear sentence boundaries
        sentence1 = "First sentence. "
        sentence2 = "Second sentence! "
        sentence3 = "Third sentence? "
        # Make it long enough to require chunking
        long_text = (sentence1 + sentence2 + sentence3) * 50  # ~4200 chars

        chunks = generator._chunk_text(long_text)

        # Should split on sentence boundaries
        assert len(chunks) >= 1
        # Each chunk should end with complete sentences (period, !, or ?)
        for chunk in chunks:
            if chunk:
                # Chunks should be reasonable length
                assert len(chunk) <= 4000

    def test_to_jsonl_format(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test JSONL export format."""
        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
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


class TestCEPGenerationPromptIntegration:
    """Tests for generation_prompt propagation through the full CEP pipeline."""

    def test_pipeline_produces_non_none_generation_prompt(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test that full pipeline produces pairs with non-None generation_prompt."""
        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        result = generator.generate_qa_pairs(sample_transcription)

        assert len(result.qa_pairs) > 0
        for pair in result.qa_pairs:
            assert pair.generation_prompt is not None
            assert len(pair.generation_prompt) > 0

    def test_jsonl_export_includes_generation_prompt(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test that JSONL export includes generation_prompt field."""
        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        result = generator.generate_qa_pairs(sample_transcription)
        jsonl = result.to_jsonl()

        lines = [line for line in jsonl.strip().split("\n") if line]
        for line in lines:
            data = json.loads(line)
            assert "generation_prompt" in data
            assert data["generation_prompt"] is not None


class TestValidationSummary:
    """Tests for validation summary including self_containedness."""

    def test_validation_summary_includes_self_containedness(
        self,
        mock_llm_client: Any,
        mock_validator_client: Any,
        qa_config: QAConfig,
        sample_transcription: EnrichedRecord,
    ) -> None:
        """Test that validation summary includes avg_self_containedness."""
        cep_config = CEPConfig(
            enable_validation=True,
            validation_threshold=0.6,
        )

        generator = CEPQAGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
            validator_client=mock_validator_client,
        )

        result = generator.generate_qa_pairs(sample_transcription)

        assert result.validation_summary is not None
        assert "avg_self_containedness" in result.validation_summary


class TestQARecordCEP:
    """Tests for QARecordCEP schema."""

    def test_bloom_distribution_validation(self) -> None:
        """Test Bloom distribution validation."""
        from arandu.qa.schemas import QAPairCEP

        pairs = [
            QAPairCEP(
                question="Q1?",
                answer="A1",
                context="C1",
                question_type="factual",
                confidence=0.9,
                bloom_level="remember",
            ),
            QAPairCEP(
                question="Q2?",
                answer="A2",
                context="C2",
                question_type="conceptual",
                confidence=0.9,
                bloom_level="analyze",
            ),
        ]

        record = QARecordCEP(
            source_file_id="test123",
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
        from arandu.qa.schemas import QAPairCEP

        pairs = [
            QAPairCEP(
                question="Q?",
                answer="A",
                context="C",
                question_type="factual",
                confidence=0.9,
                bloom_level="remember",
            ),
        ]

        record = QARecordCEP(
            source_file_id="test123",
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
        from arandu.qa.schemas import QAPairCEP

        pairs = [
            QAPairCEP(
                question="Por que?",
                answer="Porque sim.",
                context="Contexto de teste.",
                question_type="conceptual",
                confidence=0.9,
                bloom_level="analyze",
                reasoning_trace="A → B → C",
            ),
        ]

        record = QARecordCEP(
            source_file_id="test123",
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
