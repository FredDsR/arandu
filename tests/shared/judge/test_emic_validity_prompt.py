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


# Prompt *content* is asserted in test_emic_ruler.py, against ruler.pt.yaml (the
# single source shared with the annotator sheet). Duplicating content assertions
# here would let the two drift: the ruler is the thing to hold the prompt to.
