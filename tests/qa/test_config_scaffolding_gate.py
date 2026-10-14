"""Tests for the scaffolding-gate settings on CEPConfig."""

from __future__ import annotations

import pytest

from arandu.qa.config import CEPConfig, JudgeConfig, require_gate_validator


def test_gate_defaults_to_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARANDU_CEP_ENABLE_SCAFFOLDING_GATE", raising=False)
    config = CEPConfig(_env_file=None)
    assert config.enable_scaffolding_gate is True
    assert config.scaffolding_gate_active is True


def test_gate_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARANDU_CEP_ENABLE_SCAFFOLDING_GATE", "false")
    assert CEPConfig(_env_file=None).enable_scaffolding_gate is False


def test_gate_inactive_without_scaffolding_context() -> None:
    config = CEPConfig(enable_scaffolding_context=False, enable_scaffolding_gate=True)
    assert config.scaffolding_gate_active is False


def test_require_gate_validator_raises_without_model() -> None:
    cep = CEPConfig(enable_scaffolding_gate=True, enable_scaffolding_context=True)
    judge = JudgeConfig(validator_model=None)
    with pytest.raises(ValueError, match="ARANDU_JUDGE_VALIDATOR_MODEL"):
        require_gate_validator(cep, judge)


def test_require_gate_validator_accepts_model() -> None:
    cep = CEPConfig(enable_scaffolding_gate=True)
    require_gate_validator(cep, JudgeConfig(validator_model="qwen3:14b"))


def test_require_gate_validator_ignores_disabled_gate() -> None:
    cep = CEPConfig(enable_scaffolding_gate=False)
    require_gate_validator(cep, JudgeConfig(validator_model=None))


def test_gate_inactive_without_scaffolding_needs_no_validator() -> None:
    cep = CEPConfig(enable_scaffolding_context=False, enable_scaffolding_gate=True)
    require_gate_validator(cep, JudgeConfig(validator_model=None))
