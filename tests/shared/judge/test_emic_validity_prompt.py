"""Wiring tests for the emic_validity ordinal criterion prompt + config.

The emic-validity criterion is the first consumer of the ordinal judge type.
It needs no bespoke class: it is an ``OrdinalLLMCriterion`` loaded from the
``emic_validity`` prompt/config under ``prompts/judge/criteria/``. These tests
assert the on-disk prompt/config are well-formed, wire up through the generic
loader, and honor the spec's "modo antropólogo" requirements (§3, §4.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from arandu.shared.judge.criterion import OrdinalCriterionResponse, OrdinalLLMCriterion
from arandu.shared.judge.schemas import CriterionScore
from arandu.utils.paths import get_project_root

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

CRITERIA_DIR = get_project_root() / "prompts" / "judge" / "criteria"
EMIC = "emic_validity"


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    client = mocker.MagicMock()
    client.provider.value = "ollama"
    client.model_id = "test-model"
    return client


@pytest.fixture
def emic_prompt() -> str:
    return (CRITERIA_DIR / EMIC / "pt" / "prompt.md").read_text(encoding="utf-8")


class TestEmicValidityWiring:
    def test_loads_as_ordinal_criterion(self, mock_llm_client: Any) -> None:
        criterion = OrdinalLLMCriterion.from_config(
            name=EMIC,
            prompts_dir=CRITERIA_DIR,
            language="pt",
            llm_client=mock_llm_client,
        )
        assert criterion.scale == "ordinal"
        assert criterion.name == EMIC

    def test_low_temperature(self, mock_llm_client: Any) -> None:
        # The judgment is structural, not creative (spec §4.2 principle 8).
        criterion = OrdinalLLMCriterion.from_config(
            name=EMIC, prompts_dir=CRITERIA_DIR, language="pt", llm_client=mock_llm_client
        )
        assert criterion.temperature <= 0.2

    def test_evaluate_produces_ordinal_score(self, mock_llm_client: Any) -> None:
        mock_llm_client.generate_structured.return_value = OrdinalCriterionResponse(
            score=2, rationale="reenquadramento acadêmico"
        )
        criterion = OrdinalLLMCriterion.from_config(
            name=EMIC, prompts_dir=CRITERIA_DIR, language="pt", llm_client=mock_llm_client
        )
        result = criterion.evaluate(
            context="o pescador disse que eles te prejudicam",
            question="Como o pescador descreve a atuação dos órgãos?",
            answer="Há negligência sistêmica dos órgãos ambientais.",
        )
        assert isinstance(result, CriterionScore)
        assert result.scale == "ordinal"
        assert result.ordinal_score == 2
        assert mock_llm_client.generate_structured.call_args.kwargs["response_model"] is (
            OrdinalCriterionResponse
        )


class TestEmicPromptContent:
    """The prompt must encode the spec's construct, markers, anchors, blinding."""

    def test_names_construct_and_emic_etic(self, emic_prompt: str) -> None:
        low = emic_prompt.lower()
        assert "validade êmica" in low
        assert "êmic" in low and "étic" in low  # names the distinction verbatim

    def test_warns_of_model_own_reframing_bias(self, emic_prompt: str) -> None:
        # Principle 2: meta-awareness of the model's own tendency to elevate
        # situated speech into institutional/analytical diagnoses.
        low = emic_prompt.lower()
        assert "negligência sistêmica" in low  # the canonical bias example
        assert "institucion" in low or "diagnóstico" in low

    def test_generalization_is_acceptable(self, emic_prompt: str) -> None:
        # Refined construct: meaning-preserving generalization is desirable,
        # not a deviation; only meaning-change / added claims / register-swap
        # lower the score.
        low = emic_prompt.lower()
        assert "generaliz" in low
        assert "não é violação" in low or "desejável" in low

    def test_has_five_point_anchors(self, emic_prompt: str) -> None:
        # Anchors 1..5 must all appear so the model maps to the same scale.
        for level in ("1", "2", "3", "4", "5"):
            assert level in emic_prompt

    def test_only_blinded_inputs_referenced(self, emic_prompt: str) -> None:
        # Parity/blinding (§4.2 principle 7): sees only segment + Q + A.
        assert "$context" in emic_prompt
        assert "$question" in emic_prompt
        assert "$answer" in emic_prompt
        low = emic_prompt.lower()
        assert "bloom" not in low
        assert "tacit_inference" not in low and "inferência tácita" not in low

    def test_requests_structured_integer_output(self, emic_prompt: str) -> None:
        low = emic_prompt.lower()
        assert "json" in low
        assert "rationale" in low or "justificativa" in low
