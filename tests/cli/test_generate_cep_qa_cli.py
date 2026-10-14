"""Tests for the scaffolding-gate options of `arandu generate-cep-qa`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from arandu.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from an empty dir (no .env) with no gate/judge env leaking in."""
    monkeypatch.chdir(tmp_path)
    for var in (
        "ARANDU_CEP_ENABLE_SCAFFOLDING_GATE",
        "ARANDU_CEP_ENABLE_SCAFFOLDING_CONTEXT",
        "ARANDU_JUDGE_VALIDATOR_MODEL",
        "ARANDU_JUDGE_VALIDATOR_PROVIDER",
        "ARANDU_JUDGE_VALIDATOR_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    return input_dir


def test_gate_without_validator_model_exits_before_batch(
    runner: CliRunner, clean_env: Path, mocker: MockerFixture
) -> None:
    batch = mocker.patch("arandu.qa.batch.run_batch_cep_generation")

    result = runner.invoke(app, ["generate-cep-qa", str(clean_env)])

    assert result.exit_code == 1
    assert "ARANDU_JUDGE_VALIDATOR_MODEL" in result.output
    batch.assert_not_called()


def test_no_scaffolding_gate_needs_no_validator(
    runner: CliRunner, clean_env: Path, mocker: MockerFixture
) -> None:
    batch = mocker.patch("arandu.qa.batch.run_batch_cep_generation")

    result = runner.invoke(app, ["generate-cep-qa", str(clean_env), "--no-scaffolding-gate"])

    assert result.exit_code == 0, result.output
    cep_config = batch.call_args.args[3]
    assert cep_config.enable_scaffolding_gate is False


def test_validator_options_reach_the_batch(
    runner: CliRunner, clean_env: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    batch = mocker.patch("arandu.qa.batch.run_batch_cep_generation")

    result = runner.invoke(
        app,
        [
            "generate-cep-qa",
            str(clean_env),
            "--validator-model",
            "judge-model",
            "--validator-provider",
            "custom",
            "--validator-base-url",
            "http://judge/v1",
        ],
    )

    assert result.exit_code == 0, result.output
    judge_config = batch.call_args.kwargs["judge_config"]
    assert judge_config.validator_model == "judge-model"
    assert judge_config.validator_provider == "custom"
    assert judge_config.validator_base_url == "http://judge/v1"
    assert batch.call_args.args[3].enable_scaffolding_gate is True


def test_validator_model_from_env_is_enough(
    runner: CliRunner,
    clean_env: Path,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARANDU_JUDGE_VALIDATOR_MODEL", "env-judge")
    batch = mocker.patch("arandu.qa.batch.run_batch_cep_generation")

    result = runner.invoke(app, ["generate-cep-qa", str(clean_env)])

    assert result.exit_code == 0, result.output
    assert batch.call_args.kwargs["judge_config"].validator_model == "env-judge"


def test_invalid_judge_config_is_a_clean_error(
    runner: CliRunner, clean_env: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    monkeypatch.setenv("ARANDU_JUDGE_LANGUAGE", "xx")
    batch = mocker.patch("arandu.qa.batch.run_batch_cep_generation")

    result = runner.invoke(app, ["generate-cep-qa", str(clean_env), "--validator-model", "m"])

    assert result.exit_code == 1
    assert "Invalid judge configuration" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)
    batch.assert_not_called()


def test_unbuildable_validator_client_exits_before_batch(
    runner: CliRunner, clean_env: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    monkeypatch.delenv("ARANDU_LLM_BASE_URL", raising=False)
    batch = mocker.patch("arandu.qa.batch.run_batch_cep_generation")

    result = runner.invoke(
        app,
        [
            "generate-cep-qa",
            str(clean_env),
            "--validator-model",
            "judge-model",
            "--validator-provider",
            "custom",
        ],
    )

    assert result.exit_code == 1
    assert "requires a base" in result.output
    batch.assert_not_called()
