"""CLI surface for the annotation instrument."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from arandu.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
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

    def test_a_malformed_ruler_exits_one_instead_of_traceback(self, mocker: MockerFixture) -> None:
        """render_labeling_config indexes the ruler, so a missing key is a KeyError."""
        mocker.patch(
            "arandu.cli.annotation.run_build_annotation", side_effect=KeyError("loss_types")
        )
        result = runner.invoke(app, ["emic-annotation-build", "--id", "x", "--seed", "1"])
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        # print_error writes to stderr_console; the merged `.output` carries it
        # (see test_unsigned_ruler_exits_one_and_names_the_gate above).
        # Rich folds the long ruler path across lines; rejoin before matching.
        unwrapped = "".join(result.output.split())
        assert "loss_types" in unwrapped
        assert "ruler.pt.yaml" in unwrapped


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


class TestPullCommand:
    def test_is_registered(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "emic-annotation-pull" in result.stdout

    def test_file_mode_needs_no_credentials(self, mocker: MockerFixture, tmp_path: Path) -> None:
        summary = mocker.Mock(project_id=0, annotators={"A1": 3}, total_items=3, skipped=0)
        pull = mocker.patch("arandu.cli.annotation.run_pull_annotation", return_value=summary)
        settings = mocker.patch("arandu.cli.annotation.LabelStudioSettings")
        export = tmp_path / "export.json"
        export.write_text("[]", encoding="utf-8")
        result = runner.invoke(app, ["emic-annotation-pull", "--id", "run-a", "-f", str(export)])
        assert result.exit_code == 0
        settings.assert_not_called()
        assert pull.call_args.kwargs["export_file"] == export

    def test_desync_exits_one(self, mocker: MockerFixture) -> None:
        mocker.patch("arandu.cli.annotation.LabelStudioSettings")
        mocker.patch("arandu.cli.annotation.build_client_from_settings")
        mocker.patch(
            "arandu.cli.annotation.run_pull_annotation", side_effect=KeyError("task_id 99")
        )
        result = runner.invoke(app, ["emic-annotation-pull", "--id", "run-a"])
        assert result.exit_code == 1

    def test_project_id_is_threaded_through(self, mocker: MockerFixture) -> None:
        summary = mocker.Mock(project_id=42, annotators={"A1": 3}, total_items=3, skipped=0)
        mocker.patch("arandu.cli.annotation.LabelStudioSettings")
        mocker.patch("arandu.cli.annotation.build_client_from_settings")
        pull = mocker.patch("arandu.cli.annotation.run_pull_annotation", return_value=summary)
        result = runner.invoke(app, ["emic-annotation-pull", "--id", "run-a", "--project-id", "42"])
        assert result.exit_code == 0
        assert pull.call_args.kwargs["project_id"] == 42

    def test_project_id_defaults_to_none(self, mocker: MockerFixture) -> None:
        summary = mocker.Mock(project_id=42, annotators={"A1": 3}, total_items=3, skipped=0)
        mocker.patch("arandu.cli.annotation.LabelStudioSettings")
        mocker.patch("arandu.cli.annotation.build_client_from_settings")
        pull = mocker.patch("arandu.cli.annotation.run_pull_annotation", return_value=summary)
        runner.invoke(app, ["emic-annotation-pull", "--id", "run-a"])
        assert pull.call_args.kwargs["project_id"] is None

    def test_skips_are_reported(self, mocker: MockerFixture) -> None:
        summary = mocker.Mock(
            project_id=42, annotators={"A1": 2}, total_items=3, skipped=1, stale_removed=0
        )
        mocker.patch("arandu.cli.annotation.LabelStudioSettings")
        mocker.patch("arandu.cli.annotation.build_client_from_settings")
        mocker.patch("arandu.cli.annotation.run_pull_annotation", return_value=summary)
        result = runner.invoke(app, ["emic-annotation-pull", "--id", "run-a"])
        assert result.exit_code == 0
        # print_warning writes to stderr_console; the merged `.output` carries it
        # (see TestBuildCommand above).
        assert "skipped" in result.output

    def test_removed_stale_label_files_are_reported(self, mocker: MockerFixture) -> None:
        """A label file that vanishes must be visible, not silent."""
        summary = mocker.Mock(
            project_id=42, annotators={"A1": 2}, total_items=3, skipped=0, stale_removed=2
        )
        mocker.patch("arandu.cli.annotation.LabelStudioSettings")
        mocker.patch("arandu.cli.annotation.build_client_from_settings")
        mocker.patch("arandu.cli.annotation.run_pull_annotation", return_value=summary)
        result = runner.invoke(app, ["emic-annotation-pull", "--id", "run-a"])
        assert result.exit_code == 0
        unwrapped = "".join(result.output.split())
        assert "2labelfile(s)fromanearlierpullwereremoved" in unwrapped

    def test_a_file_and_a_project_id_together_exit_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The core refuses the pair; the CLI must surface it as exit 1, not ignore it."""
        from arandu.shared.annotation.build import MANIFEST_FILENAME
        from arandu.shared.annotation.schemas import AnnotationManifest

        outputs = tmp_path / "results" / "run-a" / "annotation" / "outputs"
        outputs.mkdir(parents=True)
        AnnotationManifest(
            pipeline_id="run-a",
            seed=5,
            total_items=1,
            per_cell=15,
            pool_sha256="c" * 64,
            ruler_sha256="d" * 64,
            task_map={"0": "src-a:0"},
            project_id=42,
            project_ids=[42],
        ).save(outputs / MANIFEST_FILENAME)
        monkeypatch.setenv("ARANDU_RESULTS_BASE_DIR", str(tmp_path / "results"))

        export = tmp_path / "export.json"
        export.write_text("[]", encoding="utf-8")
        result = runner.invoke(
            app,
            ["emic-annotation-pull", "--id", "run-a", "-f", str(export), "--project-id", "99"],
        )

        assert result.exit_code == 1
        unwrapped = "".join(result.output.split())
        assert "--project-id99" in unwrapped
        assert not (outputs / "labels").exists()
