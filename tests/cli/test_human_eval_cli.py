"""Tests for the ``arandu build-human-eval-sample`` command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from arandu.cli.app import app
from arandu.shared.human_eval.sampling import PER_CELL
from arandu.shared.human_eval.schemas import SampleManifest

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture

runner = CliRunner()


@pytest.fixture
def batch(mocker: MockerFixture) -> MagicMock:
    """Patch the batch so the CLI is tested without touching the filesystem.

    Uses a real ``SampleManifest`` instead of ``mocker.Mock(spec=SampleManifest)``.
    Pydantic v2 model fields are not class attributes, so ``spec=SampleManifest``
    does not constrain them: ``Mock(spec=SampleManifest, excluded_none_score=3)``
    is accepted without error even though that field no longer exists. A real
    instance gives the same protection the spec was meant to provide (the CLI
    only reads attributes off it, and a removed field still raises
    ``AttributeError``), without also accepting attributes pydantic would
    reject.
    """
    manifest = SampleManifest(
        pipeline_id="run1",
        seed=42,
        total_items=120,
        per_cell=30,
        cell_counts={"remember": 30, "understand": 30, "analyze": 30, "evaluate": 30},
        population_by_cell={"remember": 433, "understand": 644, "analyze": 644, "evaluate": 689},
        excluded_not_approved=0,
        excluded_bloom={},
        pool_sha256="0" * 64,
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
        """The pre-build echo must be detectable even if post-build output changes.

        This test invokes with --per-cell 7, which produces 7*4=28 pairs in the
        pre-build echo. The mock's total_items stays 120, so print_success cannot
        produce 28 -- only the pre-build echo can. This prevents a regression where
        removing the print_info line would go undetected because the test would
        still pass on the 120 from print_success.
        """
        result = runner.invoke(
            app,
            ["build-human-eval-sample", "--id", "run1", "--seed", "42", "--per-cell", "7"],
        )
        assert result.exit_code == 0
        assert "28" in result.stdout

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
