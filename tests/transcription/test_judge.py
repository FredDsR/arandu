"""Tests for TranscriptionJudge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from arandu.shared.judge import BaseJudge
from arandu.shared.judge.criterion import CriterionResponse
from arandu.transcription.judge import TranscriptionJudge

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# Long enough Portuguese text to exceed 30 wpm at 60s duration (>30 words)
_GOOD_PT_TEXT = (
    "O pescador mencionou a enchente que afetou a regiao no ultimo ano. "
    "Ele relatou que as aguas subiram rapidamente e inundaram as casas "
    "proximas ao rio. As familias tiveram que evacuar durante a noite "
    "e buscar abrigo em areas mais elevadas da comunidade."
)

_GOOD_EN_TEXT = (
    "The fisherman mentioned the flood that affected the region last year. "
    "He reported that the waters rose rapidly and flooded the houses near "
    "the river. The families had to evacuate during the night and seek "
    "shelter in higher areas of the community."
)


class TestTranscriptionJudge:
    def test_is_base_judge_subclass(self) -> None:
        judge = TranscriptionJudge()
        assert isinstance(judge, BaseJudge)

    def test_pipeline_has_heuristic_stage(self) -> None:
        judge = TranscriptionJudge()
        stages = judge._pipeline._stages
        assert len(stages) == 1
        assert stages[0].name == "heuristic_filter"
        assert stages[0].mode == "filter"

    def test_good_transcription_passes(self) -> None:
        judge = TranscriptionJudge()
        result = judge.evaluate_transcription(
            text=_GOOD_PT_TEXT,
            duration_ms=60000,
        )
        assert result.passed is True

    def test_cjk_text_rejected(self) -> None:
        judge = TranscriptionJudge()
        result = judge.evaluate_transcription(
            text="\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059\u3002" * 20,
            duration_ms=60000,
        )
        assert result.passed is False
        assert result.rejected_at == "heuristic_filter"

    def test_evaluate_transcription_uses_init_language(self) -> None:
        judge = TranscriptionJudge(language="en")
        result = judge.evaluate_transcription(
            text=_GOOD_EN_TEXT,
            duration_ms=60000,
        )
        assert result.passed is True

    def test_no_llm_client_needed(self) -> None:
        """TranscriptionJudge works without any LLM client."""
        judge = TranscriptionJudge()
        # Should not raise -- no LLM calls involved
        result = judge.evaluate_transcription(
            text=_GOOD_PT_TEXT,
            duration_ms=60000,
        )
        assert result.passed is True


@pytest.fixture
def mock_llm_client(mocker: MockerFixture) -> Any:
    """Create a mock LLM client that returns a passing score by default."""
    client = mocker.MagicMock()
    client.provider.value = "ollama"
    client.model_id = "test-model"
    client.generate_structured.return_value = CriterionResponse(score=1.0, rationale="clean")
    return client


class TestTranscriptionJudgeLLMStage:
    """LLM-stage pipeline assembly and evaluation behavior."""

    def test_pipeline_adds_llm_stage_when_client_provided(self, mock_llm_client: Any) -> None:
        judge = TranscriptionJudge(validator_client=mock_llm_client)
        stages = judge._pipeline._stages
        assert [s.name for s in stages] == ["heuristic_filter", "llm_filter"]
        assert stages[1].mode == "filter"

    def test_llm_stage_passes_on_clean_text(self, mock_llm_client: Any) -> None:
        judge = TranscriptionJudge(validator_client=mock_llm_client)
        result = judge.evaluate_transcription(
            text=_GOOD_PT_TEXT,
            duration_ms=60000,
        )
        assert result.passed is True
        assert "llm_filter" in result.stage_results
        llm_scores = result.stage_results["llm_filter"].criterion_scores
        assert set(llm_scores.keys()) == {"language_drift", "hallucination_loop"}
        # Two criteria => two LLM calls
        assert mock_llm_client.generate_structured.call_count == 2

    def test_llm_stage_rejects_on_drift(self, mock_llm_client: Any) -> None:
        def _structured_response(**kwargs: Any) -> CriterionResponse:
            prompt = kwargs.get("prompt", "")
            if "Language Drift" in prompt or "Linguística" in prompt:
                return CriterionResponse(score=0.0, rationale="fully English")
            return CriterionResponse(score=1.0, rationale="no hallucination")

        mock_llm_client.generate_structured.side_effect = _structured_response

        judge = TranscriptionJudge(validator_client=mock_llm_client)
        result = judge.evaluate_transcription(
            text=_GOOD_PT_TEXT,
            duration_ms=60000,
        )
        assert result.passed is False
        assert result.rejected_at == "llm_filter"
        assert result.stage_results["llm_filter"].criterion_scores["language_drift"].passed is False

    def test_llm_stage_skipped_when_heuristic_rejects(self, mock_llm_client: Any) -> None:
        judge = TranscriptionJudge(validator_client=mock_llm_client)
        result = judge.evaluate_transcription(
            text="これはテストです。" * 20,
            duration_ms=60000,
        )
        assert result.passed is False
        assert result.rejected_at == "heuristic_filter"
        assert "llm_filter" not in result.stage_results
        # LLM never called because heuristic filter short-circuited
        mock_llm_client.generate_structured.assert_not_called()

    def test_llm_criteria_receive_text_and_language(self, mock_llm_client: Any) -> None:
        judge = TranscriptionJudge(language="en", validator_client=mock_llm_client)
        judge.evaluate_transcription(text=_GOOD_EN_TEXT, duration_ms=60000)

        prompts_sent = [
            call.kwargs["prompt"] for call in mock_llm_client.generate_structured.call_args_list
        ]
        assert len(prompts_sent) == 2
        for prompt in prompts_sent:
            assert _GOOD_EN_TEXT in prompt
        # language_drift prompt must include the expected_language substitution
        drift_prompt = next(p for p in prompts_sent if "Language Drift" in p)
        assert "**Expected language:** en" in drift_prompt
