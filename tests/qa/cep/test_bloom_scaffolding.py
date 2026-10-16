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
from arandu.qa.cep.metadata_context import build_pair_judge_context
from arandu.qa.config import CEPConfig, QAConfig
from arandu.qa.schemas import QAPairCEP
from arandu.shared.judge.schemas import CriterionScore, JudgePipelineResult, JudgeStepResult
from arandu.shared.schemas import SourceMetadata
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
        temperature=0.7,
        max_tokens=2048,
    )


@pytest.fixture
def cep_config() -> CEPConfig:
    """Create a CEP config for testing (one pair per level, four levels)."""
    return CEPConfig(
        bloom_distribution={
            "remember": 1,
            "understand": 1,
            "analyze": 1,
            "evaluate": 1,
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

    def test_calculate_level_distribution_returns_configured_counts(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """The distribution is the configured per-level counts, verbatim."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        distribution = generator._calculate_level_distribution()

        # Fixture configures one pair per level across four levels
        assert distribution == {
            "remember": 1,
            "understand": 1,
            "analyze": 1,
            "evaluate": 1,
        }

    def test_calculate_level_distribution_uneven(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test level distribution with uneven counts."""
        cep_config = CEPConfig(
            bloom_distribution={
                "remember": 7,
                "understand": 3,
            },
        )

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        distribution = generator._calculate_level_distribution()

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
        pairs = generator.generate(context)

        # Fixture: 4 levels at 1 pair each -> 4 LLM calls
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

    def test_parse_response_ignores_echoed_validation(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """A 'validation' key echoed by the generation LLM never becomes a verdict."""
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )
        response = _valid_pair_json(
            validation={"stage_results": {}, "passed": True, "rejected_at": None}
        )

        pair = generator._parse_response(response, "Context.", "remember")

        assert pair.validation is None
        assert pair.is_valid is None

    def test_parse_response_drops_legacy_enrichment_fields(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        cep_config: CEPConfig,
    ) -> None:
        """Legacy reasoning-enrichment fields emitted by the model are discarded."""
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
                "hop_count": 10,
                "tacit_inference": "Conhecimento implícito X",
            }
        )

        pair = generator._parse_response(response, "Contexto de teste.", "analyze")

        assert pair.reasoning_trace is None
        assert pair.is_multi_hop is False
        assert pair.hop_count is None
        assert pair.tacit_inference is None

    def test_load_prompts_file_not_found(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        mocker: MockerFixture,
    ) -> None:
        """Test that FileNotFoundError is raised when prompt file doesn't exist."""
        cep_config = CEPConfig(
            bloom_distribution={"remember": 1},
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
            bloom_distribution={"remember": 1},
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
            bloom_distribution={
                "remember": 2,
                "understand": 0,  # Zero count
            },
            language="pt",
        )

        mock_llm_client.generate.return_value = GenerateResult(content=_valid_pair_json())

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pairs = generator.generate("Context.")

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

        pairs = generator.generate("Context.")

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
            bloom_distribution={"remember": 1},
            language="pt",
        )

        mock_llm_client.generate.return_value = GenerateResult(content=_valid_pair_json())

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        generator.generate("Context text.")

        call_kwargs = mock_llm_client.generate.call_args.kwargs
        assert "response_format" not in call_kwargs

    def test_generate_without_scaffolding_preserves_config_order(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that disabled scaffolding preserves config order, not hierarchy."""
        cep_config = CEPConfig(
            bloom_distribution={"evaluate": 2, "remember": 2},
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

        generator.generate("Context.")

        assert call_order == ["evaluate", "evaluate", "remember", "remember"]

    def test_generate_with_scaffolding_sorts_by_hierarchy(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that enabled scaffolding sorts levels by Bloom hierarchy."""
        cep_config = CEPConfig(
            bloom_distribution={"evaluate": 2, "remember": 1, "analyze": 3},
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

        generator.generate("Context.")

        # Counts evaluate=2, remember=1, analyze=3 are emitted in Bloom
        # hierarchy order: remember (1) → analyze (3) → evaluate (2)
        assert call_order == ["remember", "analyze", "analyze", "analyze", "evaluate", "evaluate"]

    def test_scaffolding_includes_prior_pairs_in_prompt(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
    ) -> None:
        """Test that each prompt includes all prior pairs (within and across levels)."""
        cep_config = CEPConfig(
            bloom_distribution={"remember": 2, "understand": 2},
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

        generator.generate("Context.")

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
            bloom_distribution={"remember": 2, "understand": 2},
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

        generator.generate("Context.")

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
            bloom_distribution={"remember": 1},
            language="pt",
        )

        mock_llm_client.generate.return_value = GenerateResult(content=_valid_pair_json())

        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client,
            qa_config=qa_config,
            cep_config=cep_config,
        )

        pairs = generator.generate("Context text.")

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
            bloom_distribution={"remember": 2, "understand": 2},
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
            bloom_distribution={"remember": 2},
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

        pairs = generator.generate("Context.")

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
            bloom_distribution={"remember": 1},
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

        pairs = generator.generate("Context.")

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
            bloom_distribution={"remember": 2},
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
            pairs = generator.generate("Context.")

        assert pairs == []
        assert "Generated 0/2 pairs for remember level" in caplog.text


class TestDefaultThesisDistribution:
    """The locked thesis-run Bloom config (2026-06-16): 3/1/1/1 integer counts.

    remember=3 is the factual base / control group + Bloom-scaffolding ground;
    understand/analyze/evaluate=1 each are the cognitive group (equal-sized
    factual-vs-cognitive split). Counts are absolute, so the per-chunk ladder
    is the sum (6) with no fractional rounding to skew the realized split.
    """

    def test_default_distribution_is_3_1_1_1(self, mock_llm_client: Any) -> None:
        cep_config = CEPConfig()  # default 3/1/1/1 integer counts
        # The per-chunk ladder size is the sum of the configured counts.
        assert sum(cep_config.bloom_distribution.values()) == 6
        generator = BloomScaffoldingGenerator(
            llm_client=mock_llm_client, qa_config=QAConfig(), cep_config=cep_config
        )
        dist = generator._calculate_level_distribution()
        assert dist == {"remember": 3, "understand": 1, "analyze": 1, "evaluate": 1}


SCAFFOLDING_HEADER = "Pares QA gerados anteriormente"


def _verdict(*, passed: bool, error: str | None = None) -> JudgePipelineResult:
    """One-criterion CEP verdict; a criterion error makes it fail like the real pipeline."""
    criterion = CriterionScore(
        score=1.0 if passed else 0.0, threshold=0.5, rationale="r", error=error
    )
    step = JudgeStepResult(criterion_scores={"faithfulness": criterion})
    return JudgePipelineResult(stage_results={"cep_validation": step}, passed=step.passed)


class _GateByQuestion:
    """PairValidator double that rejects the questions listed in ``reject``.

    ``outcome`` switches every call to an errored verdict (``"error"``), no
    verdict (``"none"``, mirrors QAJudge swallowing an exception) or a raise
    (``"raise"``).
    """

    def __init__(self, reject: set[str] | None = None, outcome: str = "verdict") -> None:
        self.reject = reject or set()
        self.outcome = outcome
        self.calls: list[tuple[QAPairCEP, str]] = []

    def validate(self, qa_pair: QAPairCEP, context: str) -> QAPairCEP:
        self.calls.append((qa_pair, context))
        if self.outcome == "raise":
            raise RuntimeError("judge down")
        if self.outcome == "none":
            return qa_pair
        error = "llm down" if self.outcome == "error" else None
        passed = qa_pair.question not in self.reject
        return qa_pair.model_copy(update={"validation": _verdict(passed=passed, error=error)})


def _numbered_llm(mock_llm_client: Any) -> list[str]:
    """Make the mock LLM answer "Question n?" / "Answer n." and capture prompts."""
    prompts: list[str] = []

    def generate(prompt: str, **kwargs: Any) -> GenerateResult:
        prompts.append(prompt)
        n = len(prompts)
        return GenerateResult(
            content=_valid_pair_json(question=f"Question {n}?", answer=f"Answer {n}.")
        )

    mock_llm_client.generate.side_effect = generate
    return prompts


def _gated(
    mock_llm_client: Any,
    qa_config: QAConfig,
    distribution: dict[str, int],
    validator: Any,
    **cep_overrides: Any,
) -> BloomScaffoldingGenerator:
    cep_config = CEPConfig(
        bloom_distribution=distribution,
        enable_scaffolding_context=True,
        language="pt",
        **cep_overrides,
    )
    return BloomScaffoldingGenerator(
        llm_client=mock_llm_client,
        qa_config=qa_config,
        cep_config=cep_config,
        validator=validator,
    )


class TestScaffoldingGate:
    """Pairs reach later prompts only after the validator approves them."""

    def test_rejected_pair_excluded_from_next_level_but_kept_in_output(
        self, mock_llm_client: Any, qa_config: QAConfig
    ) -> None:
        prompts = _numbered_llm(mock_llm_client)
        gate = _GateByQuestion(reject={"Question 1?"})
        generator = _gated(mock_llm_client, qa_config, {"remember": 1, "understand": 1}, gate)

        pairs = generator.generate("Context.")

        assert "Question 1?" not in prompts[1]
        assert [p.question for p in pairs] == ["Question 1?", "Question 2?"]
        assert pairs[0].is_valid is False
        assert pairs[1].is_valid is True

    def test_approved_pair_reaches_next_level(
        self, mock_llm_client: Any, qa_config: QAConfig
    ) -> None:
        prompts = _numbered_llm(mock_llm_client)
        generator = _gated(
            mock_llm_client, qa_config, {"remember": 1, "understand": 1}, _GateByQuestion()
        )

        generator.generate("Context.")

        assert "Question 1?" in prompts[1]

    def test_rejected_sibling_excluded_within_level(
        self, mock_llm_client: Any, qa_config: QAConfig
    ) -> None:
        prompts = _numbered_llm(mock_llm_client)
        gate = _GateByQuestion(reject={"Question 1?"})
        generator = _gated(mock_llm_client, qa_config, {"remember": 3}, gate)

        generator.generate("Context.")

        assert "Question 1?" not in prompts[2]
        assert "Question 2?" in prompts[2]

    @pytest.mark.parametrize("outcome", ["none", "error"])
    def test_missing_or_errored_verdict_not_admitted(
        self, mock_llm_client: Any, qa_config: QAConfig, outcome: str
    ) -> None:
        prompts = _numbered_llm(mock_llm_client)
        gate = _GateByQuestion(outcome=outcome)
        generator = _gated(mock_llm_client, qa_config, {"remember": 1, "understand": 1}, gate)

        pairs = generator.generate("Context.")

        assert "Question 1?" not in prompts[1]
        assert len(pairs) == 2

    def test_gate_exception_keeps_pair_unjudged(
        self, mock_llm_client: Any, qa_config: QAConfig
    ) -> None:
        prompts = _numbered_llm(mock_llm_client)
        gate = _GateByQuestion(outcome="raise")
        generator = _gated(mock_llm_client, qa_config, {"remember": 1, "understand": 1}, gate)

        pairs = generator.generate("Context.")

        assert [p.question for p in pairs] == ["Question 1?", "Question 2?"]
        assert all(p.validation is None for p in pairs)
        assert "Question 1?" not in prompts[1]

    def test_max_scaffolding_pairs_applies_after_filter(
        self, mock_llm_client: Any, qa_config: QAConfig
    ) -> None:
        prompts = _numbered_llm(mock_llm_client)
        gate = _GateByQuestion(reject={"Question 2?"})
        generator = _gated(
            mock_llm_client,
            qa_config,
            {"remember": 2, "understand": 1},
            gate,
            max_scaffolding_pairs=1,
        )

        generator.generate("Context.")

        # Cutting before filtering would keep only the rejected Question 2.
        assert "Question 1?" in prompts[2]
        assert "Question 2?" not in prompts[2]

    def test_validator_gets_the_judge_qa_grounding(
        self, mock_llm_client: Any, qa_config: QAConfig
    ) -> None:
        _numbered_llm(mock_llm_client)
        gate = _GateByQuestion()
        generator = _gated(mock_llm_client, qa_config, {"remember": 1}, gate)
        metadata = SourceMetadata(participant_name="Dona Maria", location="Barra de Pelotas")

        generator.generate("Chunk text.", source_metadata=metadata)

        expected = build_pair_judge_context(
            "Chunk text.", "Chunk text.", metadata, enable_metadata=True, language="pt"
        )
        assert gate.calls[0][1] == expected
        assert "Dona Maria" in expected

    def test_validator_receives_unjudged_pair(
        self, mock_llm_client: Any, qa_config: QAConfig
    ) -> None:
        _numbered_llm(mock_llm_client)
        gate = _GateByQuestion()
        generator = _gated(mock_llm_client, qa_config, {"remember": 2}, gate)

        generator.generate("Context.")

        assert [pair.validation for pair, _ in gate.calls] == [None, None]

    def test_approve_all_gate_matches_ungated_prompts(
        self, mocker: MockerFixture, qa_config: QAConfig
    ) -> None:
        distribution = {"remember": 2, "understand": 1, "analyze": 1}
        ungated_client = mocker.MagicMock()
        ungated_prompts = _numbered_llm(ungated_client)
        _gated(ungated_client, qa_config, distribution, None).generate("Context.")

        gated_client = mocker.MagicMock()
        gated_prompts = _numbered_llm(gated_client)
        _gated(gated_client, qa_config, distribution, _GateByQuestion()).generate("Context.")

        assert gated_prompts == ungated_prompts

    def test_empty_pool_warns_and_drops_scaffolding_section(
        self,
        mock_llm_client: Any,
        qa_config: QAConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        prompts = _numbered_llm(mock_llm_client)
        gate = _GateByQuestion(reject={"Question 1?"})
        generator = _gated(mock_llm_client, qa_config, {"remember": 1, "understand": 1}, gate)

        with caplog.at_level("WARNING", logger="arandu.qa.cep.bloom_scaffolding"):
            generator.generate("Context.")

        assert SCAFFOLDING_HEADER not in prompts[1]
        assert "No pair admitted to scaffolding before understand" in caplog.text

    def test_no_validator_means_no_judging(self, mock_llm_client: Any, qa_config: QAConfig) -> None:
        _numbered_llm(mock_llm_client)
        generator = _gated(mock_llm_client, qa_config, {"remember": 2}, None)

        pairs = generator.generate("Context.")

        assert all(p.validation is None for p in pairs)
