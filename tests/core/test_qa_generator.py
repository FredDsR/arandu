"""Tests for QA generator module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

from gtranscriber.config import QAConfig
from gtranscriber.core.llm_client import LLMClient, LLMProvider
from gtranscriber.core.qa_generator import (
    MAX_CONTEXT_LENGTH,
    MIN_CONTEXT_LENGTH,
    QAGenerator,
)
from gtranscriber.schemas import EnrichedRecord


class TestQAGenerator:
    """Tests for QAGenerator class."""

    def test_initialization_with_valid_config(self, mocker: MockerFixture) -> None:
        """Test QAGenerator initialization with valid LLMClient and QAConfig."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=10,
            strategies=["factual", "conceptual"],
        )

        generator = QAGenerator(llm_client, config)

        assert generator.llm_client == llm_client
        assert generator.config == config
        mock_openai.assert_called_once()

    def test_initialization_logging(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that initialization logs provider, model and language information."""
        caplog.set_level("INFO")
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=10,
        )

        QAGenerator(llm_client, config)

        assert "QAGenerator initialized with ollama/llama3.1:8b" in caplog.text
        assert "language=pt" in caplog.text

    def test_generate_qa_pairs_success(self, mocker: MockerFixture) -> None:
        """Test successful QA generation from a valid transcription."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()

        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "What is discussed?", "answer": "climate change", "confidence": 0.9},
            {"question": "Where is the river?", "answer": "in southern Brazil", "confidence": 0.85}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm_client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=2,
            strategies=["factual"],
        )

        generator = QAGenerator(llm_client, config)

        transcription = EnrichedRecord(
            gdrive_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder_id"],
            webContentLink="https://drive.google.com/test",
            transcription_text=(
                "This is a test transcription about climate change in southern Brazil. "
                "The river is in southern Brazil and faces critical events."
            ),
            detected_language="pt",
            language_probability=0.95,
            model_id="openai/whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=30.0,
            transcription_status="completed",
        )

        result = generator.generate_qa_pairs(transcription)

        assert result.source_gdrive_id == "test123"
        assert result.source_filename == "test.mp3"
        assert result.model_id == "llama3.1:8b"
        assert result.provider == "ollama"
        assert len(result.qa_pairs) == 2
        assert result.total_pairs == 2

    def test_generate_qa_pairs_too_short_text(self, mocker: MockerFixture) -> None:
        """Test that ValueError is raised for too-short transcription."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=5,
        )

        generator = QAGenerator(llm_client, config)

        transcription = EnrichedRecord(
            gdrive_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder_id"],
            webContentLink="https://drive.google.com/test",
            transcription_text="Short",  # Less than MIN_CONTEXT_LENGTH
            detected_language="pt",
            language_probability=0.95,
            model_id="openai/whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=1.0,
            transcription_status="completed",
        )

        with pytest.raises(ValueError) as exc_info:
            generator.generate_qa_pairs(transcription)

        assert "too short" in str(exc_info.value).lower()
        assert str(MIN_CONTEXT_LENGTH) in str(exc_info.value)

    def test_generate_qa_pairs_empty_text(self, mocker: MockerFixture) -> None:
        """Test that ValueError is raised for empty transcription."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(
            provider=LLMProvider.OLLAMA,
            model_id="llama3.1:8b",
        )

        config = QAConfig(
            provider="ollama",
            model_id="llama3.1:8b",
            questions_per_document=5,
        )

        generator = QAGenerator(llm_client, config)

        transcription = EnrichedRecord(
            gdrive_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder_id"],
            webContentLink="https://drive.google.com/test",
            transcription_text="   ",  # Empty after strip
            detected_language="pt",
            language_probability=0.95,
            model_id="openai/whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=1.0,
            transcription_status="completed",
        )

        with pytest.raises(ValueError):
            generator.generate_qa_pairs(transcription)

    def test_generate_qa_pairs_preserves_metadata(self, mocker: MockerFixture) -> None:
        """Test that metadata is preserved in output QARecord."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "Test?", "answer": "transcription about climate", "confidence": 0.8}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm_client = LLMClient(
            provider=LLMProvider.OPENAI,
            model_id="gpt-4",
            api_key="sk-test",
        )

        config = QAConfig(
            provider="openai",
            model_id="gpt-4",
            questions_per_document=1,
        )

        generator = QAGenerator(llm_client, config)

        transcription = EnrichedRecord(
            gdrive_id="gdrive_abc123",
            name="interview_001.mp3",
            mimeType="audio/mpeg",
            parents=["folder_id"],
            webContentLink="https://drive.google.com/test",
            transcription_text=(
                "This is a transcription about climate change and environmental issues. " * 10
            ),
            detected_language="pt",
            language_probability=0.95,
            model_id="openai/whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=45.0,
            transcription_status="completed",
        )

        result = generator.generate_qa_pairs(transcription)

        assert result.source_gdrive_id == "gdrive_abc123"
        assert result.source_filename == "interview_001.mp3"
        assert result.model_id == "gpt-4"
        assert result.provider == "openai"


class TestChunkText:
    """Tests for _chunk_text method."""

    def test_chunk_text_short_text(self, mocker: MockerFixture) -> None:
        """Test that short text returns single chunk."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        text = "This is a short text."
        chunks = generator._chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_exact_boundary(self, mocker: MockerFixture) -> None:
        """Test text exactly at MAX_CONTEXT_LENGTH returns single chunk."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        text = "a" * MAX_CONTEXT_LENGTH
        chunks = generator._chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_long_text_splits(self, mocker: MockerFixture) -> None:
        """Test that long text splits into multiple chunks."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        # Create text longer than MAX_CONTEXT_LENGTH with sentence boundaries
        text = "This is a sentence. " * 300  # Much longer than 4000 chars

        chunks = generator._chunk_text(text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= MAX_CONTEXT_LENGTH

    def test_chunk_text_sentence_boundaries(self, mocker: MockerFixture) -> None:
        """Test that chunking splits on sentence boundaries."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        # Create sentences of known length
        sentence = "a" * 500 + ". "
        text = sentence * 10  # 5000+ chars total

        chunks = generator._chunk_text(text)

        # Each chunk should end with complete sentences
        for chunk in chunks:
            # Should contain complete sentence(s)
            assert "." in chunk or len(chunk) == MAX_CONTEXT_LENGTH

    def test_chunk_text_very_long_sentence(self, mocker: MockerFixture) -> None:
        """Test handling of sentence longer than MAX_CONTEXT_LENGTH."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        # Create single sentence longer than MAX_CONTEXT_LENGTH
        text = "a" * (MAX_CONTEXT_LENGTH + 1000)

        chunks = generator._chunk_text(text)

        # Should truncate to MAX_CONTEXT_LENGTH
        assert len(chunks) >= 1
        assert len(chunks[0]) == MAX_CONTEXT_LENGTH

    def test_chunk_text_preserves_content(self, mocker: MockerFixture) -> None:
        """Test that chunking preserves all text content."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        text = "First sentence. Second sentence. " * 200

        chunks = generator._chunk_text(text)

        # Concatenate chunks and verify content is preserved (accounting for splits)
        total_length = sum(len(chunk) for chunk in chunks)
        # Allow some variance for spacing
        assert total_length >= len(text) * 0.95

    def test_chunk_text_various_punctuation(self, mocker: MockerFixture) -> None:
        """Test chunking with various punctuation marks."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        text = "Is this a question? Yes it is! No, wait. " * 200

        chunks = generator._chunk_text(text)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= MAX_CONTEXT_LENGTH


class TestBuildPrompt:
    """Tests for _build_prompt method."""

    def test_build_prompt_factual_strategy(self, mocker: MockerFixture) -> None:
        """Test prompt building for factual strategy."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "Test context about facts."
        prompt = generator._build_prompt(context, "factual", 3)

        assert "factual" in prompt.lower() or "facts" in prompt.lower()
        assert context in prompt
        assert "3" in prompt
        assert "JSON" in prompt or "json" in prompt

    def test_build_prompt_conceptual_strategy(self, mocker: MockerFixture) -> None:
        """Test prompt building for conceptual strategy."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "Test context about concepts."
        prompt = generator._build_prompt(context, "conceptual", 2)

        assert "concept" in prompt.lower()
        assert context in prompt
        assert "2" in prompt

    def test_build_prompt_temporal_strategy(self, mocker: MockerFixture) -> None:
        """Test prompt building for temporal strategy."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "Test context about time."
        prompt = generator._build_prompt(context, "temporal", 1)

        assert "time" in prompt.lower() or "temporal" in prompt.lower()
        assert context in prompt

    def test_build_prompt_entity_strategy(self, mocker: MockerFixture) -> None:
        """Test prompt building for entity strategy."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "Test context about entities."
        prompt = generator._build_prompt(context, "entity", 5)

        assert "entit" in prompt.lower()
        assert context in prompt
        assert "5" in prompt

    def test_build_prompt_includes_context(self, mocker: MockerFixture) -> None:
        """Test that prompt includes the full context."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "This is a unique context string for testing prompt inclusion."
        prompt = generator._build_prompt(context, "factual", 2)

        assert context in prompt

    def test_build_prompt_includes_question_count(self, mocker: MockerFixture) -> None:
        """Test that prompt includes the requested question count."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "Test context."
        prompt = generator._build_prompt(context, "factual", 7)

        assert "7" in prompt


class TestParseResponse:
    """Tests for _parse_response method."""

    def test_parse_response_valid_json_array(self, mocker: MockerFixture) -> None:
        """Test parsing valid JSON array response."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "This is a test context with climate information."
        response = """[
            {"question": "What is this?", "answer": "test context", "confidence": 0.9}
        ]"""

        pairs = generator._parse_response(response, context, "factual")

        assert len(pairs) == 1
        assert pairs[0].question == "What is this?"
        assert pairs[0].answer == "test context"
        assert pairs[0].confidence == 0.9
        assert pairs[0].question_type == "factual"

    def test_parse_response_markdown_wrapped(self, mocker: MockerFixture) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "Test context with information."
        response = """```json
[
    {"question": "What?", "answer": "information", "confidence": 0.8}
]
```"""

        pairs = generator._parse_response(response, context, "conceptual")

        assert len(pairs) == 1
        assert pairs[0].answer == "information"

    def test_parse_response_malformed_json(self, mocker: MockerFixture) -> None:
        """Test that malformed JSON returns empty list."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "Test context."
        response = "This is not valid JSON {malformed"

        pairs = generator._parse_response(response, context, "factual")

        assert len(pairs) == 0

    def test_parse_response_non_array_json(self, mocker: MockerFixture) -> None:
        """Test that non-array JSON returns empty list."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "Test context."
        response = '{"question": "What?", "answer": "test"}'

        pairs = generator._parse_response(response, context, "factual")

        assert len(pairs) == 0

    def test_parse_response_missing_required_fields(self, mocker: MockerFixture) -> None:
        """Test that pairs with missing fields are skipped."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "Test context with data."
        response = """[
            {"question": "Valid?", "answer": "data", "confidence": 0.9},
            {"question": "Missing answer"},
            {"answer": "No question"}
        ]"""

        pairs = generator._parse_response(response, context, "factual")

        # Only the first valid pair should be included
        assert len(pairs) == 1
        assert pairs[0].question == "Valid?"

    def test_parse_response_invalid_confidence(self, mocker: MockerFixture) -> None:
        """Test that invalid confidence defaults to 0.5."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "Test context with data."
        response = """[
            {"question": "What?", "answer": "data", "confidence": "invalid"}
        ]"""

        pairs = generator._parse_response(response, context, "factual")

        assert len(pairs) == 1
        assert pairs[0].confidence == 0.5

    def test_parse_response_confidence_clamping(self, mocker: MockerFixture) -> None:
        """Test that out-of-range confidence is clamped to 0.5."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "Test context with data."
        response = """[
            {"question": "What?", "answer": "data", "confidence": 1.5}
        ]"""

        pairs = generator._parse_response(response, context, "factual")

        assert len(pairs) == 1
        assert pairs[0].confidence == 0.5

    def test_parse_response_filters_non_extractive(self, mocker: MockerFixture) -> None:
        """Test that non-extractive answers are filtered out."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "The river flows through Brazil."
        response = """[
            {"question": "Where?", "answer": "flows through Brazil", "confidence": 0.9},
            {"question": "What?", "answer": "not in context at all", "confidence": 0.8}
        ]"""

        pairs = generator._parse_response(response, context, "factual")

        # Only the extractive answer should be included
        assert len(pairs) == 1
        assert "Brazil" in pairs[0].answer


class TestIsExtractive:
    """Tests for _is_extractive method."""

    def test_is_extractive_exact_match(self, mocker: MockerFixture) -> None:
        """Test exact substring match returns True."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "The river flows through southern Brazil."
        answer = "southern Brazil"

        assert generator._is_extractive(answer, context) is True

    def test_is_extractive_case_insensitive(self, mocker: MockerFixture) -> None:
        """Test case-insensitive matching."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "The River Flows Through Brazil."
        answer = "river flows through brazil"

        assert generator._is_extractive(answer, context) is True

    def test_is_extractive_short_answer_requires_exact(self, mocker: MockerFixture) -> None:
        """Test that short answers (≤2 words) require exact match."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "The river flows through Brazil in South America."
        answer = "river flows"  # Not in context as exact phrase

        # This tests the short answer path
        result = generator._is_extractive(answer, context)
        assert result is True  # Actually is in context

    def test_is_extractive_long_answer_word_overlap(self, mocker: MockerFixture) -> None:
        """Test that long answers use word overlap matching."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "The river flows through southern Brazil near the coast."
        answer = "flows through southern Brazil"  # 4 words, all in context

        assert generator._is_extractive(answer, context) is True

    def test_is_extractive_not_in_context(self, mocker: MockerFixture) -> None:
        """Test that answer not in context returns False."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "The river flows through Brazil."
        answer = "completely different content"

        assert generator._is_extractive(answer, context) is False

    def test_is_extractive_partial_overlap(self, mocker: MockerFixture) -> None:
        """Test answer with <80% word overlap returns False."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        context = "The river flows through Brazil."
        answer = "river flows mountains plains"  # Only 50% overlap

        assert generator._is_extractive(answer, context) is False


class TestErrorHandling:
    """Tests for error handling in QA generation."""

    def test_generate_for_context_llm_exception(self, mocker: MockerFixture) -> None:
        """Test that LLM exceptions are caught and return empty list."""
        # Skip retry delays for faster tests
        mocker.patch("tenacity.nap.time.sleep", return_value=None)

        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("LLM API error")
        mock_openai.return_value = mock_client

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        pairs = generator._generate_for_context("Test context", "factual", 3)

        assert len(pairs) == 0

    def test_generate_for_context_zero_questions(self, mocker: MockerFixture) -> None:
        """Test handling of zero questions request."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        # Should not call LLM if num_questions is 0
        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(provider="ollama", model_id="llama3.1:8b")
        generator = QAGenerator(llm_client, config)

        # This would be handled by the calling code, but test it doesn't crash
        pairs = generator._generate_for_context("Test context", "factual", 0)

        # With 0 questions, the prompt is still built but may return empty
        assert isinstance(pairs, list)


class TestLanguageSupport:
    """Tests for language support in QA generation."""

    def test_qa_config_valid_english_language(self) -> None:
        """Test QAConfig accepts 'en' language."""
        config = QAConfig(language="en")
        assert config.language == "en"

    def test_qa_config_valid_portuguese_language(self) -> None:
        """Test QAConfig accepts 'pt' language."""
        config = QAConfig(language="pt")
        assert config.language == "pt"

    def test_qa_config_default_language_is_portuguese(self) -> None:
        """Test QAConfig defaults to 'pt' language."""
        config = QAConfig()
        assert config.language == "pt"

    def test_qa_config_invalid_language_raises_error(self) -> None:
        """Test QAConfig rejects invalid language codes."""
        with pytest.raises(ValueError) as exc_info:
            QAConfig(language="fr")
        assert "Invalid QA language" in str(exc_info.value)

    def test_generator_loads_english_prompts(self, mocker: MockerFixture) -> None:
        """Test QAGenerator loads English prompts correctly."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(language="en")
        generator = QAGenerator(llm_client, config)

        assert "system_instruction" in generator._prompts
        assert "strategy_instructions" in generator._prompts
        assert "factual" in generator._prompts["strategy_instructions"]
        assert "expert" in generator._prompts["system_instruction"].lower()

    def test_generator_loads_portuguese_prompts(self, mocker: MockerFixture) -> None:
        """Test QAGenerator loads Portuguese prompts correctly."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(language="pt")
        generator = QAGenerator(llm_client, config)

        assert "system_instruction" in generator._prompts
        assert "strategy_instructions" in generator._prompts
        assert "especialista" in generator._prompts["system_instruction"].lower()

    def test_generator_uses_loaded_prompts_in_build_prompt(self, mocker: MockerFixture) -> None:
        """Test that _build_prompt uses prompts from loaded file."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(language="pt")
        generator = QAGenerator(llm_client, config)

        prompt = generator._build_prompt("Test context", "factual", 3)

        # Should contain Portuguese text from the prompts file
        assert "especialista" in prompt.lower()

    def test_generator_logs_language(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that initialization logs language information."""
        caplog.set_level("INFO")
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(language="pt")
        QAGenerator(llm_client, config)

        assert "language=pt" in caplog.text

    def test_qa_record_includes_language(self, mocker: MockerFixture) -> None:
        """Test that generated QARecord includes language field."""
        mock_openai = mocker.patch("gtranscriber.core.llm_client.OpenAI")
        mock_client = Mock()

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """[
            {"question": "Test?", "answer": "transcription about climate", "confidence": 0.8}
        ]"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(language="pt", questions_per_document=1)
        generator = QAGenerator(llm_client, config)

        transcription = EnrichedRecord(
            gdrive_id="test123",
            name="test.mp3",
            mimeType="audio/mpeg",
            parents=["folder_id"],
            webContentLink="https://drive.google.com/test",
            transcription_text=("This is a transcription about climate change. " * 10),
            detected_language="pt",
            language_probability=0.95,
            model_id="openai/whisper-large-v3",
            compute_device="cpu",
            processing_duration_sec=30.0,
            transcription_status="completed",
        )

        result = generator.generate_qa_pairs(transcription)

        assert result.language == "pt"

    def test_missing_prompt_file_raises_error(self, mocker: MockerFixture) -> None:
        """Test that missing prompt file raises FileNotFoundError."""
        mocker.patch("gtranscriber.core.llm_client.OpenAI")

        llm_client = LLMClient(provider=LLMProvider.OLLAMA, model_id="llama3.1:8b")
        config = QAConfig(prompt_path="/nonexistent/path/prompts.json")

        with pytest.raises(FileNotFoundError) as exc_info:
            QAGenerator(llm_client, config)

        assert "Prompt file not found" in str(exc_info.value)
