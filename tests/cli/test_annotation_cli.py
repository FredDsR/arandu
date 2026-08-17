"""CLI surface for the annotation instrument."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from arandu.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

runner = CliRunner()


class TestBuildCommand:
    def test_is_registered(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "emic-annotation-build" in result.stdout

    def test_missing_sample_exits_one(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "arandu.cli.annotation.run_build_annotation",
            side_effect=FileNotFoundError("no sample"),
        )
        result = runner.invoke(app, ["emic-annotation-build", "--id", "x", "--seed", "1"])
        assert result.exit_code == 1

    def test_unsigned_ruler_exits_one_and_names_the_gate(self, mocker: MockerFixture) -> None:
        from arandu.shared.annotation.ruler import RulerNotSignedOffError

        mocker.patch(
            "arandu.cli.annotation.run_build_annotation",
            side_effect=RulerNotSignedOffError("gate anthropologist-validation is open"),
        )
        result = runner.invoke(app, ["emic-annotation-build", "--id", "x", "--seed", "1"])
        assert result.exit_code == 1
        # print_error writes to stderr_console; CliRunner keeps stdout/stderr
        # separate on this click version, so the merged `.output` is what
        # carries the message (matches the convention in the other CLI test
        # modules, e.g. tests/cli/test_kg_build_retriever_index_cli.py).
        assert "anthropologist-validation" in result.output

    def test_success_reports_the_output_dir(self, mocker: MockerFixture, tmp_path: Path) -> None:
        manifest = mocker.Mock(total_items=120, seed=9, project_id=None)
        mocker.patch("arandu.cli.annotation.run_build_annotation", return_value=manifest)
        result = runner.invoke(app, ["emic-annotation-build", "--id", "run-a", "--seed", "9"])
        assert result.exit_code == 0
        assert "annotation/outputs" in result.stdout


class TestPushCommand:
    def test_is_registered(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "emic-annotation-push" in result.stdout

    def test_missing_credentials_exit_one(self, mocker: MockerFixture) -> None:
        mocker.patch(
            "arandu.cli.annotation.LabelStudioSettings",
            side_effect=ValueError("token is required"),
        )
        result = runner.invoke(app, ["emic-annotation-push", "--id", "run-a"])
        assert result.exit_code == 1
        # print_error writes to stderr_console; CliRunner keeps stdout/stderr
        # separate on this click version, so the merged `.output` is what
        # carries the message (see TestBuildCommand above).
        assert "ARANDU_LABEL_STUDIO_TOKEN" in result.output

    def test_duplicate_push_exits_one(self, mocker: MockerFixture) -> None:
        mocker.patch("arandu.cli.annotation.LabelStudioSettings")
        mocker.patch("arandu.cli.annotation.build_client_from_settings")
        mocker.patch(
            "arandu.cli.annotation.run_push_annotation",
            side_effect=ValueError("already pushed as project 42"),
        )
        result = runner.invoke(app, ["emic-annotation-push", "--id", "run-a"])
        assert result.exit_code == 1
