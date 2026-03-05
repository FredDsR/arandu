"""Tests for Reasoning and Grounding Enrichment module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from arandu.core.cep.reasoning import ReasoningEnricher
from arandu.qa.config import CEPConfig
from arandu.qa.schemas import QAPairCEP
from arandu.utils.text import GenerateResult

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    """Create a mock LLM client."""
    return mocker.MagicMock()


@pytest.fixture
def cep_config() -> CEPConfig:
    """Create a CEP config for testing."""
    return CEPConfig(
        enable_reasoning_traces=True,
        max_hop_count=5,
        language="pt",
    )


@pytest.fixture
def sample_qa_pair_analyze() -> QAPairCEP:
    """Create a sample QA pair at analyze level."""
    return QAPairCEP(
        question="Por que o pescador guarda o barco quando o rio sobe?",
        answer="Para evitar perda do equipamento.",
        context="Se o rio sobe rápido, guardo o barco para evitar perda.",
        question_type="conceptual",
        confidence=0.9,
        bloom_level="analyze",
    )


@pytest.fixture
def sample_qa_pair_remember() -> QAPairCEP:
    """Create a sample QA pair at remember level."""
    return QAPairCEP(
        question="Qual é o nome do rio mencionado?",
        answer="Rio Amazonas.",
        context="O Rio Amazonas é o maior rio do Brasil.",
        question_type="factual",
        confidence=0.95,
        bloom_level="remember",
    )


class TestReasoningEnricher:
    """Tests for ReasoningEnricher class."""

    def test_initialization(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test enricher initialization."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        assert enricher.llm_client == mock_llm_client
        assert enricher.cep_config == cep_config

    def test_enrich_skips_when_disabled(
        self,
        mock_llm_client: Any,
        sample_qa_pair_analyze: QAPairCEP,
    ) -> None:
        """Test that enrichment is skipped when reasoning traces are disabled."""
        cep_config = CEPConfig(enable_reasoning_traces=False)
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = enricher.enrich(sample_qa_pair_analyze, "context")

        # Should return the same pair unchanged
        assert result == sample_qa_pair_analyze
        mock_llm_client.generate.assert_not_called()

    def test_enrich_skips_lower_levels(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair_remember: QAPairCEP,
    ) -> None:
        """Test that enrichment is skipped for remember/understand levels."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = enricher.enrich(sample_qa_pair_remember, "context")

        # Should return the same pair unchanged
        assert result == sample_qa_pair_remember
        mock_llm_client.generate.assert_not_called()

    def test_enrich_skips_existing_reasoning(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test that enrichment is skipped if pair already has reasoning trace."""
        pair_with_reasoning = QAPairCEP(
            question="Por que?",
            answer="Porque sim.",
            context="Contexto.",
            question_type="conceptual",
            confidence=0.9,
            bloom_level="analyze",
            reasoning_trace="Existing reasoning trace",
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = enricher.enrich(pair_with_reasoning, "context")

        assert result == pair_with_reasoning
        mock_llm_client.generate.assert_not_called()

    def test_enrich_calls_llm_for_analyze_level(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair_analyze: QAPairCEP,
    ) -> None:
        """Test that enrichment calls LLM for analyze level."""
        mock_llm_client.generate.return_value = GenerateResult(
            content=json.dumps(
                {
                    "reasoning_trace": "Fato A \u2192 A\u00e7\u00e3o B \u2192 Resultado C",
                    "is_multi_hop": True,
                    "hop_count": 2,
                    "tacit_inference": "O pescador sabe que enchentes causam preju\u00edzos",
                }
            )
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = enricher.enrich(
            sample_qa_pair_analyze,
            "Se o rio sobe r\u00e1pido, guardo o barco para evitar perda.",
        )

        assert mock_llm_client.generate.called
        assert result.reasoning_trace == "Fato A \u2192 A\u00e7\u00e3o B \u2192 Resultado C"
        assert result.is_multi_hop is True
        assert result.hop_count == 2
        assert result.tacit_inference == "O pescador sabe que enchentes causam preju\u00edzos"

    def test_enrich_handles_evaluate_level(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test that enrichment works for evaluate level."""
        pair = QAPairCEP(
            question="A decis\u00e3o foi acertada?",
            answer="Sim, foi prudente.",
            context="Ele decidiu guardar o barco.",
            question_type="conceptual",
            confidence=0.85,
            bloom_level="evaluate",
        )

        mock_llm_client.generate.return_value = GenerateResult(
            content=json.dumps(
                {
                    "reasoning_trace": "Decis\u00e3o \u2192 Avalia\u00e7\u00e3o positiva",
                    "is_multi_hop": False,
                    "hop_count": None,
                    "tacit_inference": "Prud\u00eancia \u00e9 valorizada na comunidade",
                }
            )
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = enricher.enrich(pair, "context")

        assert result.bloom_level == "evaluate"
        assert result.reasoning_trace == "Decis\u00e3o \u2192 Avalia\u00e7\u00e3o positiva"

    def test_enrich_handles_create_level(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test that enrichment works for create level."""
        pair = QAPairCEP(
            question="Como melhorar a prote\u00e7\u00e3o?",
            answer="Construir um abrigo elevado.",
            context="Os barcos ficam vulner\u00e1veis durante enchentes.",
            question_type="conceptual",
            confidence=0.8,
            bloom_level="create",
        )

        mock_llm_client.generate.return_value = GenerateResult(
            content=json.dumps(
                {
                    "reasoning_trace": "Problema \u2192 Solu\u00e7\u00e3o criativa",
                    "is_multi_hop": True,
                    "hop_count": 3,
                    "tacit_inference": None,
                }
            )
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = enricher.enrich(pair, "context")

        assert result.bloom_level == "create"
        assert result.reasoning_trace == "Problema \u2192 Solu\u00e7\u00e3o criativa"

    def test_enrich_does_not_pass_response_format(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair_analyze: QAPairCEP,
    ) -> None:
        """Test that enrich does not pass response_format to LLM client."""
        mock_llm_client.generate.return_value = GenerateResult(
            content=json.dumps(
                {
                    "reasoning_trace": "A \u2192 B",
                    "is_multi_hop": False,
                    "hop_count": None,
                    "tacit_inference": None,
                }
            )
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        enricher.enrich(sample_qa_pair_analyze, "context")

        call_kwargs = mock_llm_client.generate.call_args.kwargs
        assert "response_format" not in call_kwargs

    def test_enrich_handles_thinking_tags(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair_analyze: QAPairCEP,
    ) -> None:
        """Test that enrichment parses from GenerateResult.content correctly."""
        mock_llm_client.generate.return_value = GenerateResult(
            content=json.dumps(
                {
                    "reasoning_trace": "A \u2192 B",
                    "is_multi_hop": False,
                    "hop_count": None,
                    "tacit_inference": None,
                }
            ),
            thinking="internal reasoning about enrichment",
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = enricher.enrich(sample_qa_pair_analyze, "context")

        assert result.reasoning_trace == "A \u2192 B"

    def test_parse_reasoning_response_valid(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test parsing valid reasoning response."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        response = json.dumps(
            {
                "reasoning_trace": "A + B \u2192 C",
                "is_multi_hop": True,
                "hop_count": 2,
                "tacit_inference": "Implicit knowledge",
            }
        )

        result = enricher._parse_reasoning_response(response)

        assert result["reasoning_trace"] == "A + B \u2192 C"
        assert result["is_multi_hop"] is True
        assert result["hop_count"] == 2
        assert result["tacit_inference"] == "Implicit knowledge"

    def test_parse_reasoning_response_with_markdown(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test parsing response wrapped in markdown code block."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        response = """```json
{
    "reasoning_trace": "Trace",
    "is_multi_hop": false,
    "hop_count": null,
    "tacit_inference": null
}
```"""

        result = enricher._parse_reasoning_response(response)

        assert result["reasoning_trace"] == "Trace"
        assert result["is_multi_hop"] is False

    def test_parse_reasoning_response_invalid_json(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test parsing invalid JSON returns empty dict."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        response = "not valid json {"

        result = enricher._parse_reasoning_response(response)

        assert result == {}

    def test_parse_reasoning_validates_hop_count(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test that hop_count is validated against max_hop_count."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        # hop_count exceeds max_hop_count (5)
        response = json.dumps(
            {
                "reasoning_trace": "Trace",
                "is_multi_hop": True,
                "hop_count": 10,
            }
        )

        result = enricher._parse_reasoning_response(response)

        # hop_count should be None since it exceeds max
        assert result["hop_count"] is None

    def test_enrich_batch(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair_analyze: QAPairCEP,
        sample_qa_pair_remember: QAPairCEP,
    ) -> None:
        """Test batch enrichment of multiple QA pairs."""
        mock_llm_client.generate.return_value = GenerateResult(
            content=json.dumps(
                {
                    "reasoning_trace": "Trace",
                    "is_multi_hop": False,
                    "hop_count": None,
                    "tacit_inference": None,
                }
            )
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        pairs = [sample_qa_pair_analyze, sample_qa_pair_remember]
        results = enricher.enrich_batch(pairs, "context")

        assert len(results) == 2
        # Only analyze pair should have been enriched
        assert results[0].reasoning_trace == "Trace"
        assert results[1].reasoning_trace is None

    def test_enrich_preserves_generation_prompt(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test that generation_prompt is preserved through enrich()."""
        pair = QAPairCEP(
            question="Por que?",
            answer="Porque sim.",
            context="Contexto.",
            question_type="conceptual",
            confidence=0.9,
            bloom_level="analyze",
            generation_prompt="Original prompt",
        )

        mock_llm_client.generate.return_value = GenerateResult(
            content=json.dumps(
                {
                    "reasoning_trace": "A \u2192 B",
                    "is_multi_hop": False,
                    "hop_count": None,
                    "tacit_inference": None,
                }
            )
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = enricher.enrich(pair, "context")

        assert result.generation_prompt == "Original prompt"

    def test_enrich_handles_llm_error(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
        sample_qa_pair_analyze: QAPairCEP,
    ) -> None:
        """Test that enrichment handles LLM errors gracefully."""
        mock_llm_client.generate.side_effect = Exception("LLM error")

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        result = enricher.enrich(sample_qa_pair_analyze, "context")

        # Should return original pair on error
        assert result == sample_qa_pair_analyze

    def test_load_prompts_file_not_found(
        self,
        mock_llm_client: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test that FileNotFoundError is raised when prompt file doesn't exist."""
        cep_config = CEPConfig(
            enable_reasoning_traces=True,
            language="pt",
        )

        # Mock the file existence check to return False
        mocker.patch("pathlib.Path.exists", return_value=False)

        with pytest.raises(FileNotFoundError, match="CEP data file not found"):
            ReasoningEnricher(
                llm_client=mock_llm_client,
                cep_config=cep_config,
            )

    def test_load_prompts_template_not_found(
        self,
        mock_llm_client: Any,
        mocker: MockerFixture,
    ) -> None:
        """Test that FileNotFoundError is raised when template file doesn't exist."""
        cep_config = CEPConfig(
            enable_reasoning_traces=True,
            language="pt",
        )

        mocker.patch("pathlib.Path.exists", side_effect=[True, False])

        with pytest.raises(FileNotFoundError, match="CEP template file not found"):
            ReasoningEnricher(
                llm_client=mock_llm_client,
                cep_config=cep_config,
            )

    def test_parse_reasoning_validates_hop_count_when_not_multi_hop(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test that hop_count is set to None when is_multi_hop is False."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        # is_multi_hop is False, so hop_count should be ignored
        response = json.dumps(
            {
                "reasoning_trace": "Trace",
                "is_multi_hop": False,
                "hop_count": 3,  # This should be ignored
            }
        )

        result = enricher._parse_reasoning_response(response)

        assert result["is_multi_hop"] is False
        # hop_count should be None when is_multi_hop is False
        assert result["hop_count"] is None

    def test_parse_reasoning_response_invalid_hop_count_type(
        self,
        mock_llm_client: Any,
        cep_config: CEPConfig,
    ) -> None:
        """Test that invalid hop_count type (e.g., string) is handled."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            cep_config=cep_config,
        )

        # hop_count is a string instead of int
        response = json.dumps(
            {
                "reasoning_trace": "Trace",
                "is_multi_hop": True,
                "hop_count": "not_a_number",
            }
        )

        result = enricher._parse_reasoning_response(response)

        assert result["is_multi_hop"] is True
        # hop_count should be None due to invalid type
        assert result["hop_count"] is None
