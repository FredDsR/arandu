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
        # 8192 headroom: the answerer emits structured output and reasoning
        # models burn thinking tokens against this budget; 1024 truncated the
        # JSON -> spurious abstention (dry-run 2026-06-08).
        assert s.max_tokens == 8192
        assert s.language == "pt"
        assert s.top_k == 10
        # 16384 (not 8192): the packer reserves max_tokens (8192) from this
        # budget, so it must leave room for passages (dry-run bug #7 follow-on).
        assert s.max_context_tokens == 16384
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

    def test_rejects_nonpositive_packing_budget(self) -> None:
        # max_context_tokens must leave room after reserving max_tokens +
        # prompt_overhead; otherwise pack_passages would crash mid-run. The
        # model_validator fails fast at construction. 8192 - 350 - 8192 < 0.
        with pytest.raises(ValueError, match="packing budget is non-positive"):
            AnswererSettings(_env_file=None, max_context_tokens=8192)

    def test_accepts_budget_with_room_for_passages(self) -> None:
        # Just enough headroom: 8543 - 350 - 8192 = 1 > 0.
        s = AnswererSettings(_env_file=None, max_context_tokens=8543)
        assert s.max_context_tokens == 8543


class TestWorkersField:
    def test_workers_defaults_to_one(self) -> None:
        assert AnswererSettings(provider="ollama").workers == 1

    def test_workers_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_ANSWERER_WORKERS", "3")
        assert AnswererSettings().workers == 3

    def test_workers_bounds(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AnswererSettings(workers=0)
        with pytest.raises(ValidationError):
            AnswererSettings(workers=17)
