"""Tests for Reasoning and Grounding Enrichment module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from gtranscriber.config import PECConfig
from gtranscriber.core.pec.reasoning import ReasoningEnricher
from gtranscriber.schemas import QAPairPEC

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    """Create a mock LLM client."""
    return mocker.MagicMock()


@pytest.fixture
def pec_config() -> PECConfig:
    """Create a PEC config for testing."""
    return PECConfig(
        enable_reasoning_traces=True,
        max_hop_count=5,
        language="pt",
    )


@pytest.fixture
def sample_qa_pair_analyze() -> QAPairPEC:
    """Create a sample QA pair at analyze level."""
    return QAPairPEC(
        question="Por que o pescador guarda o barco quando o rio sobe?",
        answer="Para evitar perda do equipamento.",
        context="Se o rio sobe rápido, guardo o barco para evitar perda.",
        question_type="conceptual",
        confidence=0.9,
        bloom_level="analyze",
    )


@pytest.fixture
def sample_qa_pair_remember() -> QAPairPEC:
    """Create a sample QA pair at remember level."""
    return QAPairPEC(
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
        pec_config: PECConfig,
    ) -> None:
        """Test enricher initialization."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
        )

        assert enricher.llm_client == mock_llm_client
        assert enricher.pec_config == pec_config

    def test_enrich_skips_when_disabled(
        self,
        mock_llm_client: Any,
        sample_qa_pair_analyze: QAPairPEC,
    ) -> None:
        """Test that enrichment is skipped when reasoning traces are disabled."""
        pec_config = PECConfig(enable_reasoning_traces=False)
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
        )

        result = enricher.enrich(sample_qa_pair_analyze, "context")

        # Should return the same pair unchanged
        assert result == sample_qa_pair_analyze
        mock_llm_client.generate.assert_not_called()

    def test_enrich_skips_lower_levels(
        self,
        mock_llm_client: Any,
        pec_config: PECConfig,
        sample_qa_pair_remember: QAPairPEC,
    ) -> None:
        """Test that enrichment is skipped for remember/understand levels."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
        )

        result = enricher.enrich(sample_qa_pair_remember, "context")

        # Should return the same pair unchanged
        assert result == sample_qa_pair_remember
        mock_llm_client.generate.assert_not_called()

    def test_enrich_skips_existing_reasoning(
        self,
        mock_llm_client: Any,
        pec_config: PECConfig,
    ) -> None:
        """Test that enrichment is skipped if pair already has reasoning trace."""
        pair_with_reasoning = QAPairPEC(
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
            pec_config=pec_config,
        )

        result = enricher.enrich(pair_with_reasoning, "context")

        assert result == pair_with_reasoning
        mock_llm_client.generate.assert_not_called()

    def test_enrich_calls_llm_for_analyze_level(
        self,
        mock_llm_client: Any,
        pec_config: PECConfig,
        sample_qa_pair_analyze: QAPairPEC,
    ) -> None:
        """Test that enrichment calls LLM for analyze level."""
        mock_llm_client.generate.return_value = json.dumps(
            {
                "reasoning_trace": "Fato A → Ação B → Resultado C",
                "is_multi_hop": True,
                "hop_count": 2,
                "tacit_inference": "O pescador sabe que enchentes causam prejuízos",
            }
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
        )

        result = enricher.enrich(
            sample_qa_pair_analyze,
            "Se o rio sobe rápido, guardo o barco para evitar perda.",
        )

        assert mock_llm_client.generate.called
        assert result.reasoning_trace == "Fato A → Ação B → Resultado C"
        assert result.is_multi_hop is True
        assert result.hop_count == 2
        assert result.tacit_inference == "O pescador sabe que enchentes causam prejuízos"

    def test_enrich_handles_evaluate_level(
        self,
        mock_llm_client: Any,
        pec_config: PECConfig,
    ) -> None:
        """Test that enrichment works for evaluate level."""
        pair = QAPairPEC(
            question="A decisão foi acertada?",
            answer="Sim, foi prudente.",
            context="Ele decidiu guardar o barco.",
            question_type="conceptual",
            confidence=0.85,
            bloom_level="evaluate",
        )

        mock_llm_client.generate.return_value = json.dumps(
            {
                "reasoning_trace": "Decisão → Avaliação positiva",
                "is_multi_hop": False,
                "hop_count": None,
                "tacit_inference": "Prudência é valorizada na comunidade",
            }
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
        )

        result = enricher.enrich(pair, "context")

        assert result.bloom_level == "evaluate"
        assert result.reasoning_trace == "Decisão → Avaliação positiva"

    def test_enrich_handles_create_level(
        self,
        mock_llm_client: Any,
        pec_config: PECConfig,
    ) -> None:
        """Test that enrichment works for create level."""
        pair = QAPairPEC(
            question="Como melhorar a proteção?",
            answer="Construir um abrigo elevado.",
            context="Os barcos ficam vulneráveis durante enchentes.",
            question_type="conceptual",
            confidence=0.8,
            bloom_level="create",
        )

        mock_llm_client.generate.return_value = json.dumps(
            {
                "reasoning_trace": "Problema → Solução criativa",
                "is_multi_hop": True,
                "hop_count": 3,
                "tacit_inference": None,
            }
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
        )

        result = enricher.enrich(pair, "context")

        assert result.bloom_level == "create"
        assert result.reasoning_trace == "Problema → Solução criativa"

    def test_parse_reasoning_response_valid(
        self,
        mock_llm_client: Any,
        pec_config: PECConfig,
    ) -> None:
        """Test parsing valid reasoning response."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
        )

        response = json.dumps(
            {
                "reasoning_trace": "A + B → C",
                "is_multi_hop": True,
                "hop_count": 2,
                "tacit_inference": "Implicit knowledge",
            }
        )

        result = enricher._parse_reasoning_response(response)

        assert result["reasoning_trace"] == "A + B → C"
        assert result["is_multi_hop"] is True
        assert result["hop_count"] == 2
        assert result["tacit_inference"] == "Implicit knowledge"

    def test_parse_reasoning_response_with_markdown(
        self,
        mock_llm_client: Any,
        pec_config: PECConfig,
    ) -> None:
        """Test parsing response wrapped in markdown code block."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
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
        pec_config: PECConfig,
    ) -> None:
        """Test parsing invalid JSON returns empty dict."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
        )

        response = "not valid json {"

        result = enricher._parse_reasoning_response(response)

        assert result == {}

    def test_parse_reasoning_validates_hop_count(
        self,
        mock_llm_client: Any,
        pec_config: PECConfig,
    ) -> None:
        """Test that hop_count is validated against max_hop_count."""
        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
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
        pec_config: PECConfig,
        sample_qa_pair_analyze: QAPairPEC,
        sample_qa_pair_remember: QAPairPEC,
    ) -> None:
        """Test batch enrichment of multiple QA pairs."""
        mock_llm_client.generate.return_value = json.dumps(
            {
                "reasoning_trace": "Trace",
                "is_multi_hop": False,
                "hop_count": None,
                "tacit_inference": None,
            }
        )

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
        )

        pairs = [sample_qa_pair_analyze, sample_qa_pair_remember]
        results = enricher.enrich_batch(pairs, "context")

        assert len(results) == 2
        # Only analyze pair should have been enriched
        assert results[0].reasoning_trace == "Trace"
        assert results[1].reasoning_trace is None

    def test_enrich_handles_llm_error(
        self,
        mock_llm_client: Any,
        pec_config: PECConfig,
        sample_qa_pair_analyze: QAPairPEC,
    ) -> None:
        """Test that enrichment handles LLM errors gracefully."""
        mock_llm_client.generate.side_effect = Exception("LLM error")

        enricher = ReasoningEnricher(
            llm_client=mock_llm_client,
            pec_config=pec_config,
        )

        result = enricher.enrich(sample_qa_pair_analyze, "context")

        # Should return original pair on error
        assert result == sample_qa_pair_analyze
