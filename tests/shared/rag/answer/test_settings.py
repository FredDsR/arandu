"""Tests for ``AnswererSettings`` — env-var parsing + provider normalization."""

from __future__ import annotations

import pytest

from arandu.shared.rag.answer.settings import AnswererSettings


class TestAnswererSettings:
    def test_defaults(self) -> None:
        # Defaults match the spec §5.7: ollama provider, qwen3:14b, low
        # temperature for deterministic answers, Portuguese prompt.
        s = AnswererSettings(_env_file=None)
        assert s.provider == "ollama"
        assert s.model_id == "qwen3:14b"
        assert s.temperature == 0.2
        assert s.max_tokens == 1024
        assert s.language == "pt"
        assert s.top_k == 10
        assert s.max_context_tokens == 8192
        assert s.prompt_overhead_tokens == 350

    def test_provider_lowercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Env-var case doesn't trip up LLMProvider() dispatch.
        monkeypatch.setenv("ARANDU_ANSWERER_PROVIDER", "OpenAI")
        assert AnswererSettings().provider == "openai"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_ANSWERER_MODEL_ID", "gpt-4o-mini")
        monkeypatch.setenv("ARANDU_ANSWERER_TEMPERATURE", "0.5")
        monkeypatch.setenv("ARANDU_ANSWERER_LANGUAGE", "en")
        s = AnswererSettings()
        assert s.model_id == "gpt-4o-mini"
        assert s.temperature == 0.5
        assert s.language == "en"

    def test_temperature_bounds(self) -> None:
        with pytest.raises(ValueError):
            AnswererSettings(temperature=-0.1)
        with pytest.raises(ValueError):
            AnswererSettings(temperature=2.5)
