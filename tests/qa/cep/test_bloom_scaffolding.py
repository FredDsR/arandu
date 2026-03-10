"""Tests for Bloom Scaffolding QA Generator."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from arandu.qa.cep.bloom_scaffolding import (
    BloomScaffoldingGenerator,
    LLMResponseError,
)
from arandu.qa.config import CEPConfig, QAConfig
from arandu.qa.schemas import QAPairCEP
from arandu.utils.text import GenerateResult

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


def _valid_pair_json(**overrides: Any) -> str:
    """Build a valid single-pair JSON string with optional overrides."""
    data = {"question": "Q?", "answer": "A.", "confidence": 0.9}
    data.update(overrides)
    return json.dumps(data)


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
        """Test that generate calls LLM once per question (one pair per call)."""
        mock_llm_client.generate.return_value = GenerateResult(
            content=_valid_pair_json(
                question="O que aconteceu?",
                answer="Uma resposta.",
            )
        )

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        context = "Este é um texto de contexto para teste."
        pairs = generator.generate(context, num_questions=4)

        # With 4 questions distributed across 4 levels (1 each), should call LLM 4 times
        # (tenacity retries don't add extra calls when responses are valid)
        assert mock_llm_client.generate.call_count == 4
        assert len(pairs) == 4

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
            {
                "question": "Qual é a capital do Brasil?",
                "answer": "Brasília",
                "confidence": 0.95,
            }
        )

        context = "A capital do Brasil é Brasília."
        pair = generator._parse_response(response, context, "remember")

        assert pair.question == "Qual é a capital do Brasil?"
        assert pair.answer == "Brasília"
        assert pair.bloom_level == "remember"
        assert pair.confidence == 0.95

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
    {
        "question": "O que é um teste?",
        "answer": "Uma verificação.",
        "confidence": 0.8
    }
```"""

        context = "Um teste é uma verificação."
        pair = generator._parse_response(response, context, "understand")

        assert pair.question == "O que é um teste?"

    def test_parse_response_invalid_json_raises(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test parsing invalid JSON response raises LLMResponseError."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        with pytest.raises(LLMResponseError, match="Invalid JSON"):
            generator._parse_response("Invalid JSON { not valid", "Context.", "remember")

    def test_parse_response_empty_question_raises(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that empty question field raises LLMResponseError."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps({"question": "", "answer": "A.", "confidence": 0.9})

        with pytest.raises(LLMResponseError, match="Validation failed"):
            generator._parse_response(response, "Context.", "remember")

    def test_parse_response_invalid_confidence_raises(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that out-of-range confidence raises LLMResponseError."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps({"question": "Q?", "answer": "A.", "confidence": 1.5})

        with pytest.raises(LLMResponseError, match="Validation failed"):
            generator._parse_response(response, "Context.", "remember")

    def test_parse_response_non_dict_raises(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that non-dict JSON raises LLMResponseError."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps([{"question": "Q?", "answer": "A."}])

        with pytest.raises(LLMResponseError, match="Expected JSON object"):
            generator._parse_response(response, "Context.", "remember")

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
            {
                "question": "Por que isso acontece?",
                "answer": "Por causa de X.",
                "confidence": 0.85,
                "reasoning_trace": "Fato A + Fato B → Conclusão",
                "is_multi_hop": True,
                "hop_count": 2,
                "tacit_inference": "Conhecimento implícito X",
            }
        )

        context = "Contexto de teste."
        pair = generator._parse_response(response, context, "analyze")

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
        """Test that hop_count outside 1-5 raises validation error."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps(
            {"question": "Q?", "answer": "A.", "is_multi_hop": True, "hop_count": 10}
        )

        with pytest.raises(LLMResponseError, match="Validation failed"):
            generator._parse_response(response, "Context.", "analyze")

    def test_parse_response_valid_hop_count(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that valid hop_count within 1-5 is accepted."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps(
            {
                "question": "Q?",
                "answer": "A.",
                "confidence": 0.9,
                "is_multi_hop": True,
                "hop_count": 3,
            }
        )

        pair = generator._parse_response(response, "Context.", "analyze")
        assert pair.hop_count == 3

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

        mock_llm_client.generate.return_value = GenerateResult(content=_valid_pair_json())

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pairs = generator.generate("Context.", num_questions=2)

        # Only 'remember' level should be called (2 times), not 'understand'
        assert mock_llm_client.generate.call_count == 2
        for pair in pairs:
            assert pair.bloom_level == "remember"

    def test_generate_for_level_handles_exception(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that exception in _generate_for_level is handled gracefully."""
        mock_llm_client.generate.side_effect = RuntimeError("LLM error")

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pairs = generator.generate("Context.", num_questions=2)

        assert pairs == []

    def test_parse_response_parses_single_object(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that a flat JSON object is parsed as a single pair."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = json.dumps({"question": "Q?", "answer": "A.", "confidence": 0.8})

        pair = generator._parse_response(response, "Context.", "remember")

        assert pair.question == "Q?"

    def test_generate_does_not_pass_response_format(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that generate does not pass response_format to LLM client."""
        cep_config = CEPConfig(
            bloom_levels=["remember"],
            bloom_distribution={"remember": 1.0},
            language="pt",
        )

        mock_llm_client.generate.return_value = GenerateResult(content=_valid_pair_json())

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        generator.generate("Context text.", num_questions=1)

        call_kwargs = mock_llm_client.generate.call_args.kwargs
        assert "response_format" not in call_kwargs

    def test_generate_without_scaffolding_preserves_config_order(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that disabled scaffolding preserves config order, not hierarchy."""
        cep_config = CEPConfig(
            bloom_levels=["evaluate", "remember"],
            bloom_distribution={"evaluate": 0.5, "remember": 0.5},
            enable_scaffolding_context=False,
            language="pt",
        )

        call_order: list[str] = []

        def track_calls(prompt: str, **kwargs: Any) -> GenerateResult:
            match = re.search(r"Nível Cognitivo: (\w+)", prompt)
            if match:
                call_order.append(match.group(1).lower())
            return GenerateResult(content=_valid_pair_json())

        mock_llm_client.generate.side_effect = track_calls

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        generator.generate("Context.", num_questions=4)

        assert call_order == ["evaluate", "evaluate", "remember", "remember"]

    def test_generate_with_scaffolding_sorts_by_hierarchy(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that enabled scaffolding sorts levels by Bloom hierarchy."""
        cep_config = CEPConfig(
            bloom_levels=["evaluate", "remember", "analyze"],
            bloom_distribution={"evaluate": 0.34, "remember": 0.33, "analyze": 0.33},
            enable_scaffolding_context=True,
            language="pt",
        )

        call_order: list[str] = []

        def track_calls(prompt: str, **kwargs: Any) -> GenerateResult:
            match = re.search(r"Nível Cognitivo: (\w+)", prompt)
            if match:
                call_order.append(match.group(1).lower())
            return GenerateResult(content=_valid_pair_json())

        mock_llm_client.generate.side_effect = track_calls

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        generator.generate("Context.", num_questions=6)

        # With 6 questions and distribution 0.34, 0.33, 0.33:
        # Using int() truncation (floor): evaluate=2, remember=1, analyze=3 (remaining)
        # Should follow hierarchy: remember (1) → analyze (3) → evaluate (2)
        assert call_order == ["remember", "analyze", "analyze", "analyze", "evaluate", "evaluate"]

    def test_scaffolding_includes_prior_pairs_in_prompt(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that each prompt includes all prior pairs (within and across levels)."""
        cep_config = CEPConfig(
            bloom_levels=["remember", "understand"],
            bloom_distribution={"remember": 0.5, "understand": 0.5},
            enable_scaffolding_context=True,
            language="pt",
        )

        prompts_captured: list[str] = []
        call_count = 0

        def capture_prompts(prompt: str, **kwargs: Any) -> GenerateResult:
            nonlocal call_count
            prompts_captured.append(prompt)
            call_count += 1
            return GenerateResult(
                content=_valid_pair_json(
                    question=f"Question {call_count}?",
                    answer=f"Answer {call_count}.",
                )
            )

        mock_llm_client.generate.side_effect = capture_prompts

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        generator.generate("Context.", num_questions=4)

        # With 4 questions: 2 remember, 2 understand = 4 LLM calls
        assert len(prompts_captured) == 4

        # Second remember call should contain first remember pair
        assert "[REMEMBER]" in prompts_captured[1]
        assert "Question 1?" in prompts_captured[1]

        # First understand call should contain both remember pairs
        assert "[REMEMBER]" in prompts_captured[2]
        assert "Question 1?" in prompts_captured[2]
        assert "Question 2?" in prompts_captured[2]

        # Second understand call should contain all prior pairs
        assert "Question 1?" in prompts_captured[3]
        assert "Question 2?" in prompts_captured[3]
        assert "Question 3?" in prompts_captured[3]

    def test_scaffolding_first_level_has_no_prior_context(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that the first pair has no scaffolding header."""
        cep_config = CEPConfig(
            bloom_levels=["remember", "understand"],
            bloom_distribution={"remember": 0.5, "understand": 0.5},
            enable_scaffolding_context=True,
            language="pt",
        )

        prompts_captured: list[str] = []

        def capture_prompts(prompt: str, **kwargs: Any) -> GenerateResult:
            prompts_captured.append(prompt)
            return GenerateResult(content=_valid_pair_json())

        mock_llm_client.generate.side_effect = capture_prompts

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        generator.generate("Context.", num_questions=4)

        # First prompt (first remember pair) should not contain scaffolding header
        scaffolding_header = generator._prompts["scaffolding_header"]
        assert scaffolding_header not in prompts_captured[0]

    def test_format_prior_pairs_empty_list(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that _format_prior_pairs returns empty string for empty input."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        result = generator._format_prior_pairs([])
        assert result == ""

    def test_format_prior_pairs_includes_bloom_level(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that output contains [REMEMBER], question, and answer."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pairs = [
            QAPairCEP(
                question="Who was affected?",
                answer="Riverine communities.",
                context="Context.",
                question_type="factual",
                confidence=0.9,
                bloom_level="remember",
            ),
        ]

        result = generator._format_prior_pairs(pairs)
        assert "[REMEMBER]" in result
        assert "Who was affected?" in result
        assert "Riverine communities." in result
        assert result.startswith("1.")

    def test_parse_response_threads_generation_prompt(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that _parse_response threads generation_prompt to pairs."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        response = _valid_pair_json()
        pair = generator._parse_response(
            response, "Context.", "remember", generation_prompt="The prompt"
        )

        assert pair.generation_prompt == "The prompt"

    def test_parse_response_defaults_generation_prompt_to_none(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that _parse_response defaults generation_prompt to None."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pair = generator._parse_response(_valid_pair_json(), "Context.", "remember")

        assert pair.generation_prompt is None

    def test_generate_sets_generation_prompt(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that generate() produces pairs with non-None generation_prompt."""
        cep_config = CEPConfig(
            bloom_levels=["remember"],
            bloom_distribution={"remember": 1.0},
            language="pt",
        )

        mock_llm_client.generate.return_value = GenerateResult(content=_valid_pair_json())

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pairs = generator.generate("Context text.", num_questions=1)

        assert len(pairs) >= 1
        for pair in pairs:
            assert pair.generation_prompt is not None
            assert len(pair.generation_prompt) > 0

    def test_format_prior_pairs_respects_max_limit(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that only last N pairs are included when exceeding limit."""
        cep_config = CEPConfig(
            bloom_levels=["remember", "understand"],
            bloom_distribution={"remember": 0.5, "understand": 0.5},
            max_scaffolding_pairs=3,
            language="pt",
        )

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pairs = [
            QAPairCEP(
                question=f"Q{i}?",
                answer=f"A{i}.",
                context="Context.",
                question_type="factual",
                confidence=0.9,
                bloom_level="remember",
            )
            for i in range(5)
        ]

        result = generator._format_prior_pairs(pairs)

        # Should only include the last 3 pairs (Q2, Q3, Q4)
        assert "Q0?" not in result
        assert "Q1?" not in result
        assert "Q2?" in result
        assert "Q3?" in result
        assert "Q4?" in result

    # =========================================================================
    # A4: generation_thinking tests
    # =========================================================================

    def test_generation_thinking_stored_per_pair(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that GenerateResult.thinking is stored in QAPairCEP.generation_thinking."""
        cep_config = CEPConfig(
            bloom_levels=["remember"],
            bloom_distribution={"remember": 1.0},
            language="pt",
        )

        mock_llm_client.generate.return_value = GenerateResult(
            content=_valid_pair_json(),
            thinking="reasoning about the question",
        )

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pairs = generator.generate("Context.", num_questions=2)

        assert len(pairs) == 2
        for pair in pairs:
            assert pair.generation_thinking == "reasoning about the question"

    def test_generation_thinking_none_when_no_thinking(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that generation_thinking is None when model has no thinking."""
        cep_config = CEPConfig(
            bloom_levels=["remember"],
            bloom_distribution={"remember": 1.0},
            language="pt",
        )

        mock_llm_client.generate.return_value = GenerateResult(
            content=_valid_pair_json(),
            thinking=None,
        )

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pairs = generator.generate("Context.", num_questions=1)

        assert len(pairs) == 1
        assert pairs[0].generation_thinking is None

    def test_parse_response_threads_thinking(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that _parse_response threads generation_thinking to the pair."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pair = generator._parse_response(
            _valid_pair_json(),
            "Context.",
            "remember",
            generation_thinking="trace text",
        )

        assert pair.generation_thinking == "trace text"

    # =========================================================================
    # B1: Retry behavior tests
    # =========================================================================

    def test_generate_single_pair_retries_on_invalid_json(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that invalid JSON triggers retry and succeeds on second attempt."""
        mock_llm_client.generate.side_effect = [
            GenerateResult(content="not valid json {"),
            GenerateResult(content=_valid_pair_json()),
        ]

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        with patch.object(
            generator._generate_single_pair.retry,
            "wait",
            return_value=0,  # type: ignore[union-attr]
        ):
            pair = generator._generate_single_pair(
                prompt="test prompt", context="Context.", bloom_level="remember"
            )

        assert pair.question == "Q?"
        assert mock_llm_client.generate.call_count == 2

    def test_generate_single_pair_retries_on_validation_error(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that Pydantic validation failure triggers retry."""
        mock_llm_client.generate.side_effect = [
            # First call: empty question (fails min_length=1)
            GenerateResult(content=json.dumps({"question": "", "answer": "A.", "confidence": 0.9})),
            # Second call: valid
            GenerateResult(content=_valid_pair_json()),
        ]

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        with patch.object(
            generator._generate_single_pair.retry,
            "wait",
            return_value=0,  # type: ignore[union-attr]
        ):
            pair = generator._generate_single_pair(
                prompt="test prompt", context="Context.", bloom_level="remember"
            )

        assert pair.question == "Q?"
        assert mock_llm_client.generate.call_count == 2

    def test_generate_single_pair_skips_after_retries_exhausted(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Test that pair is skipped after all retries are exhausted."""
        mock_llm_client.generate.return_value = GenerateResult(content="always invalid json {")

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        with (
            patch.object(
                generator._generate_single_pair.retry,
                "wait",
                return_value=0,  # type: ignore[union-attr]
            ),
            pytest.raises(LLMResponseError, match="Invalid JSON"),
        ):
            generator._generate_single_pair(
                prompt="test prompt", context="Context.", bloom_level="remember"
            )

        # 3 attempts (initial + 2 retries)
        assert mock_llm_client.generate.call_count == 3

    def test_generate_skips_failed_pairs_and_logs_summary(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that failed pairs are skipped and a summary warning is logged."""
        cep_config = CEPConfig(
            bloom_levels=["remember"],
            bloom_distribution={"remember": 1.0},
            language="pt",
        )

        # Always return invalid JSON to exhaust retries
        mock_llm_client.generate.return_value = GenerateResult(content="invalid json")

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        with patch.object(
            generator._generate_single_pair.retry,
            "wait",
            return_value=0,  # type: ignore[union-attr]
        ):
            pairs = generator.generate("Context.", num_questions=2)

        assert pairs == []
        assert "Generated 0/2 pairs for remember level" in caplog.text
