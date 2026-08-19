"""Tests for the ``arandu build-human-eval-sample`` command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from arandu.cli.app import app
from arandu.shared.human_eval.sampling import PER_CELL

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture

runner = CliRunner()


@pytest.fixture
def batch(mocker: MockerFixture) -> MagicMock:
    """Patch the batch so the CLI is tested without touching the filesystem."""
    manifest = mocker.Mock(
        total_items=120,
        per_cell=30,
        cell_counts={"remember": 30, "understand": 30, "analyze": 30, "evaluate": 30},
        excluded_not_approved=0,
        excluded_bloom={},
    )
    return mocker.patch("arandu.cli.human_eval.run_build_sample_batch", return_value=manifest)


class TestBuildHumanEvalSample:
    def test_is_registered(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "build-human-eval-sample" in result.stdout

    def test_per_cell_defaults_to_the_module_constant(self, batch: MagicMock) -> None:
        result = runner.invoke(app, ["build-human-eval-sample", "--id", "run1", "--seed", "42"])
        assert result.exit_code == 0
        assert batch.call_args.kwargs["per_cell"] == PER_CELL

    def test_per_cell_is_forwarded(self, batch: MagicMock) -> None:
        result = runner.invoke(
            app,
            ["build-human-eval-sample", "--id", "run1", "--seed", "42", "--per-cell", "10"],
        )
        assert result.exit_code == 0
        assert batch.call_args.kwargs["per_cell"] == 10

    def test_per_cell_rejects_zero(self, batch: MagicMock) -> None:
        result = runner.invoke(
            app,
            ["build-human-eval-sample", "--id", "run1", "--seed", "42", "--per-cell", "0"],
        )
        assert result.exit_code != 0
        batch.assert_not_called()

    def test_echoes_the_resulting_size_before_building(self, batch: MagicMock) -> None:
        result = runner.invoke(app, ["build-human-eval-sample", "--id", "run1", "--seed", "42"])
        assert "120" in result.stdout

    def test_insufficient_cell_exits_one(self, batch: MagicMock) -> None:
        """Only the exit code is asserted: `print_error` writes to `stderr_console`.

        Same shape as `test_desync_exits_one` in `tests/cli/test_annotation_cli.py`.
        Asserting the message on `result.stdout` only works while Click's
        `mix_stderr` default folds the streams together, which Click 8.2 drops.
        """
        batch.side_effect = ValueError("Bloom cell 'evaluate' has only 4 approved pair(s)")
        result = runner.invoke(app, ["build-human-eval-sample", "--id", "run1", "--seed", "42"])
        assert result.exit_code == 1
        assert batch.called
