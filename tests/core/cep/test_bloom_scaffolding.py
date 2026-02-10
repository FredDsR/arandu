"""Tests for Bloom Scaffolding QA Generator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from gtranscriber.config import CEPConfig, QAConfig
from gtranscriber.core.cep.bloom_scaffolding import BloomScaffoldingGenerator

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    """Create a mock LLM client."""
    client = mocker.MagicMock()
    return client


@pytest.fixture
def qa_config() -> QAConfig:
    """Create a QA config for testing."""
    return QAConfig(
        questions_per_document=10,
        temperature=0.7,
        max_tokens=2048,
    )


@pytest.fixture
def cep_config() -> CEPConfig:
    """Create a CEP config for testing."""
    return CEPConfig(
        bloom_levels=["remember", "understand", "analyze", "evaluate"],
        bloom_distribution={
            "remember": 0.25,
            "understand": 0.25,
            "analyze": 0.25,
            "evaluate": 0.25,
        },
        language="pt",
    )


class TestBloomScaffoldingGenerator:
    """Tests for BloomScaffoldingGenerator class."""

    def test_initialization(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test generator initialization."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        assert generator.llm_client == mock_llm_client
        assert generator.qa_config == qa_config
        assert generator.cep_config == cep_config

    def test_calculate_level_distribution_equal(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test level distribution calculation with equal weights."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        # With 4 levels and 10 questions (25% each)
        distribution = generator._calculate_level_distribution(8)

        # Each level should get 2 questions (8 * 0.25 = 2)
        assert distribution["remember"] == 2
        assert distribution["understand"] == 2
        assert distribution["analyze"] == 2
        # Last level gets remaining
        assert distribution["evaluate"] == 2

    def test_calculate_level_distribution_uneven(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test level distribution with uneven weights."""
        cep_config = CEPConfig(
            bloom_levels=["remember", "understand"],
            bloom_distribution={
                "remember": 0.7,
                "understand": 0.3,
            },
        )

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        distribution = generator._calculate_level_distribution(10)

        # remember: 10 * 0.7 = 7, understand gets remaining = 3
        assert distribution["remember"] == 7
        assert distribution["understand"] == 3

    def test_generate_calls_llm_for_each_level(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that generate calls LLM for each Bloom level."""
        # Setup mock to return valid JSON response
        mock_response = json.dumps(
            [
                {
                    "question": "O que aconteceu?",
                    "answer": "Uma resposta.",
                    "bloom_level": "remember",
                    "confidence": 0.9,
                }
            ]
        )
        mock_llm_client.generate.return_value = mock_response

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        context = "Este é um texto de contexto para teste."
        pairs = generator.generate(context, num_questions=4)

        # Should call LLM for each level with non-zero count
        assert mock_llm_client.generate.call_count == 4
        # Should return pairs from all levels
        assert len(pairs) >= 1

    def test_parse_response_valid_json(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test parsing valid JSON response."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps(
            [
                {
                    "question": "Qual é a capital do Brasil?",
                    "answer": "Brasília",
                    "bloom_level": "remember",
                    "confidence": 0.95,
                }
            ]
        )

        context = "A capital do Brasil é Brasília."
        pairs = generator._parse_response(response, context, "remember")

        assert len(pairs) == 1
        assert pairs[0].question == "Qual é a capital do Brasil?"
        assert pairs[0].answer == "Brasília"
        assert pairs[0].bloom_level == "remember"
        assert pairs[0].confidence == 0.95

    def test_parse_response_with_markdown_code_block(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test parsing response wrapped in markdown code block."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = """```json
[
    {
        "question": "O que é um teste?",
        "answer": "Uma verificação.",
        "bloom_level": "understand",
        "confidence": 0.8
    }
]
```"""

        context = "Um teste é uma verificação."
        pairs = generator._parse_response(response, context, "understand")

        assert len(pairs) == 1
        assert pairs[0].question == "O que é um teste?"

    def test_parse_response_invalid_json(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test parsing invalid JSON response returns empty list."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = "Invalid JSON { not valid"
        context = "Some context."
        pairs = generator._parse_response(response, context, "remember")

        assert pairs == []

    def test_parse_response_skips_invalid_items(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that invalid items in response are skipped."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps(
            [
                {"question": "", "answer": "A"},  # Empty question
                {"question": "Q", "answer": ""},  # Empty answer
                {"question": "Valid?", "answer": "Valid", "confidence": 0.9},  # Valid
            ]
        )

        context = "Context."
        pairs = generator._parse_response(response, context, "remember")

        assert len(pairs) == 1
        assert pairs[0].question == "Valid?"

    def test_parse_response_normalizes_confidence(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that confidence values outside [0, 1] are normalized."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps(
            [
                {"question": "Q1?", "answer": "A1", "confidence": 1.5},
                {"question": "Q2?", "answer": "A2", "confidence": -0.5},
                {"question": "Q3?", "answer": "A3", "confidence": "invalid"},
            ]
        )

        context = "Context."
        pairs = generator._parse_response(response, context, "remember")

        # Out of range confidence should be normalized to 0.5
        assert all(p.confidence == 0.5 for p in pairs)

    def test_bloom_to_question_type_mapping(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test Bloom level to question type mapping."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        assert generator._bloom_to_question_type("remember") == "factual"
        assert generator._bloom_to_question_type("understand") == "conceptual"
        assert generator._bloom_to_question_type("apply") == "conceptual"
        assert generator._bloom_to_question_type("analyze") == "conceptual"
        assert generator._bloom_to_question_type("evaluate") == "conceptual"
        assert generator._bloom_to_question_type("create") == "conceptual"
        assert generator._bloom_to_question_type("unknown") == "factual"

    def test_parse_response_with_cep_fields(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test parsing response with CEP-specific fields."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps(
            [
                {
                    "question": "Por que isso acontece?",
                    "answer": "Por causa de X.",
                    "bloom_level": "analyze",
                    "confidence": 0.85,
                    "reasoning_trace": "Fato A + Fato B → Conclusão",
                    "is_multi_hop": True,
                    "hop_count": 2,
                    "tacit_inference": "Conhecimento implícito X",
                }
            ]
        )

        context = "Contexto de teste."
        pairs = generator._parse_response(response, context, "analyze")

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.reasoning_trace == "Fato A + Fato B → Conclusão"
        assert pair.is_multi_hop is True
        assert pair.hop_count == 2
        assert pair.tacit_inference == "Conhecimento implícito X"

    def test_parse_response_validates_hop_count(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that hop_count is validated to be within 1-5."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps(
            [
                {"question": "Q1?", "answer": "A1", "is_multi_hop": True, "hop_count": 0},
                {"question": "Q2?", "answer": "A2", "is_multi_hop": True, "hop_count": 10},
                {"question": "Q3?", "answer": "A3", "is_multi_hop": True, "hop_count": 3},
                {"question": "Q4?", "answer": "A4", "hop_count": 2},  # No is_multi_hop
            ]
        )

        context = "Context."
        pairs = generator._parse_response(response, context, "analyze")

        # hop_count 0 and 10 are outside valid range (1-5), should default to 2 when is_multi_hop
        assert pairs[0].hop_count == 2  # Invalid 0 -> defaults to 2
        assert pairs[1].hop_count == 2  # Invalid 10 -> defaults to 2
        assert pairs[2].hop_count == 3  # Valid range
        # Q4 has hop_count but is_multi_hop defaults to False, so hop_count is set to None
        assert pairs[3].hop_count is None

    def test_load_prompts_file_not_found(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        mocker: MockerFixture,
    ) -> None:
        """Test that FileNotFoundError is raised when prompt file doesn't exist."""
        cep_config = CEPConfig(
            bloom_levels=["remember"],
            bloom_distribution={"remember": 1.0},
            language="pt",
        )

        # Mock the file existence check to return False
        mocker.patch("pathlib.Path.exists", return_value=False)

        with pytest.raises(FileNotFoundError, match="CEP data file not found"):
            BloomScaffoldingGenerator(
                llm_client=mock_llm_client,
                qa_config=qa_config,
                cep_config=cep_config,
            )

    def test_load_prompts_template_not_found(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        mocker: MockerFixture,
    ) -> None:
        """Test that FileNotFoundError is raised when template file doesn't exist."""
        cep_config = CEPConfig(
            bloom_levels=["remember"],
            bloom_distribution={"remember": 1.0},
            language="pt",
        )

        mocker.patch("pathlib.Path.exists", side_effect=[True, False])

        with pytest.raises(FileNotFoundError, match="CEP template file not found"):
            BloomScaffoldingGenerator(
                llm_client=mock_llm_client,
                qa_config=qa_config,
                cep_config=cep_config,
            )

    def test_generate_with_zero_count_level(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test generation when a Bloom level has zero questions allocated."""
        cep_config = CEPConfig(
            bloom_levels=["remember", "understand"],
            bloom_distribution={
                "remember": 1.0,
                "understand": 0.0,  # Zero weight
            },
            language="pt",
        )

        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "question": "Test?",
                    "answer": "Answer.",
                    "bloom_level": "remember",
                    "confidence": 0.9,
                }
            ]
        )

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pairs = generator.generate("Context.", num_questions=2)

        # Only 'remember' level should be called, not 'understand'
        # (since understand has count=0)
        assert mock_llm_client.generate.call_count == 1
        for pair in pairs:
            assert pair.bloom_level == "remember"

    def test_generate_for_level_handles_exception(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that exception in _generate_for_level is handled gracefully."""
        # Make the LLM client raise an exception
        mock_llm_client.generate.side_effect = Exception("LLM error")

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pairs = generator.generate("Context.", num_questions=2)

        # Should return empty list when all generations fail
        assert pairs == []

    def test_parse_response_with_non_list_data(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that non-list JSON response returns empty list."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        # Return a JSON object instead of array
        response = json.dumps({"question": "Q?", "answer": "A"})
        context = "Context."
        pairs = generator._parse_response(response, context, "remember")

        assert pairs == []

    def test_parse_response_with_non_dict_items(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that non-dict items in array are skipped."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps(
            [
                "not a dict",
                {"question": "Valid?", "answer": "Valid", "confidence": 0.9},
                123,  # Number
                None,  # None
            ]
        )

        context = "Context."
        pairs = generator._parse_response(response, context, "remember")

        # Only the valid dict should be parsed
        assert len(pairs) == 1
        assert pairs[0].question == "Valid?"

    def test_parse_response_invalid_hop_count_type(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that invalid hop_count type (e.g., string) is handled."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps(
            [
                {
                    "question": "Q1?",
                    "answer": "A1",
                    "is_multi_hop": True,
                    "hop_count": "not_a_number",  # Invalid type
                }
            ]
        )

        context = "Context."
        pairs = generator._parse_response(response, context, "analyze")

        # Should default to 2 when is_multi_hop=True and hop_count is invalid
        assert len(pairs) == 1
        assert pairs[0].hop_count == 2  # Default value from validator

    def test_parse_response_normalizes_invalid_values(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that invalid values are normalized gracefully."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        # Parser should normalize invalid values:
        # - invalid confidence -> 0.5 (default)
        # - bloom_level from JSON is ignored, uses method parameter
        response = json.dumps(
            [
                {
                    "question": "Valid?",
                    "answer": "Valid",
                    "confidence": "not_a_number",  # Invalid type -> normalized to 0.5
                    "bloom_level": "invalid_level",  # Ignored, uses parameter
                }
            ]
        )

        context = "Context."
        pairs = generator._parse_response(response, context, "remember")

        # Parser is lenient: normalizes invalid values instead of skipping
        assert len(pairs) == 1
        assert pairs[0].confidence == 0.5  # Normalized from invalid
        assert pairs[0].bloom_level == "remember"  # Uses method parameter
