"""Tests for ``JudgeAnswersSettings`` — env-var parsing + provider normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arandu.shared.rag.judge_answers.settings import JudgeAnswersSettings

if TYPE_CHECKING:
    import pytest


class TestJudgeAnswersSettings:
    def test_defaults(self) -> None:
        # Defaults: ollama / qwen3:14b / pt / temp 0.1 (low for reproducible
        # single-shot pointwise scoring, matching the rest of the judges).
        s = JudgeAnswersSettings(_env_file=None)
        assert s.provider == "ollama"
        assert s.model_id == "qwen3:14b"
        assert s.temperature == 0.1
        assert s.max_tokens == 8192
        assert s.language == "pt"
        assert s.abstention_disagreement_audit is True

    def test_provider_lowercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_JUDGE_ANSWERS_PROVIDER", "Ollama")
        assert JudgeAnswersSettings().provider == "ollama"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_JUDGE_ANSWERS_MODEL_ID", "gpt-4o-mini")
        monkeypatch.setenv("ARANDU_JUDGE_ANSWERS_LANGUAGE", "en")
        monkeypatch.setenv("ARANDU_JUDGE_ANSWERS_ABSTENTION_DISAGREEMENT_AUDIT", "false")
        s = JudgeAnswersSettings()
        assert s.model_id == "gpt-4o-mini"
        assert s.language == "en"
        assert s.abstention_disagreement_audit is False
