"""Tests for TranscriptionJudge."""

from __future__ import annotations

from arandu.shared.judge import BaseJudge
from arandu.transcription.judge import TranscriptionJudge

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
