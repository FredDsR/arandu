"""Tests for `arandu chunk` CLI."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from arandu.cli.app import app
from arandu.shared.chunking.schemas import ChunkSet
from arandu.shared.schemas import PipelineType, RunStatus

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def results_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ResultsManager base dir to a temp path for the duration of one test."""
    base = tmp_path / "results"
    monkeypatch.setenv("ARANDU_RESULTS_BASE_DIR", str(base))
    return base


def _write_enriched_record(dir_: Path, file_id: str, text: str) -> Path:
    """Write a minimal-but-valid EnrichedRecord JSON to ``dir_/file_id.json``."""
    payload = {
        "gdrive_id": file_id,
        "name": f"{file_id}.mp4",
        "mimeType": "video/mp4",
        "parents": ["folder"],
        "webContentLink": "https://drive.google.com/test",
        "size_bytes": 1024,
        "duration_milliseconds": 60000,
        "transcription_text": text,
        "detected_language": "pt",
        "language_probability": 0.95,
        "model_id": "whisper-large-v3",
        "compute_device": "cpu",
        "processing_duration_sec": 30.5,
        "transcription_status": "completed",
    }
    path = dir_ / f"{file_id}.json"
    path.write_text(json.dumps(payload))
    return path


class TestAranduChunkPathLayout:
    """Outputs land under `results/<pipeline_id>/chunk/outputs/<chunker_id>/<file_id>.json`."""

    def test_writes_chunkset_per_source_under_chunker_subdir(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_enriched_record(in_dir, "src_a", "Esta é uma frase de teste. " * 50)
        _write_enriched_record(in_dir, "src_b", "Outra frase. " * 30)

        result = runner.invoke(app, ["chunk", str(in_dir), "--id", "run_x", "--view", "cep_4k"])
        assert result.exit_code == 0, result.output

        outputs = results_base / "run_x" / "chunk" / "outputs" / "cep_4k"
        assert (outputs / "src_a.json").exists()
        assert (outputs / "src_b.json").exists()

        cs = ChunkSet.load(outputs / "src_a.json")
        assert cs.source_file_id == "src_a"
        assert set(cs.views) == {"cep_4k"}
        assert len(cs.view("cep_4k")) >= 1

    def test_multiple_views_emit_one_file_per_view(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_enriched_record(in_dir, "src_a", "Esta é uma frase de teste. " * 200)

        result = runner.invoke(
            app,
            [
                "chunk",
                str(in_dir),
                "--id",
                "run_x",
                "--view",
                "cep_4k",
                "--view",
                "nx_2k",
            ],
        )
        assert result.exit_code == 0, result.output

        cep_path = results_base / "run_x" / "chunk" / "outputs" / "cep_4k" / "src_a.json"
        nx_path = results_base / "run_x" / "chunk" / "outputs" / "nx_2k" / "src_a.json"
        assert cep_path.exists()
        assert nx_path.exists()

        # Each emitted ChunkSet carries exactly its own view, not the union.
        assert set(ChunkSet.load(cep_path).views) == {"cep_4k"}
        assert set(ChunkSet.load(nx_path).views) == {"nx_2k"}

    def test_auto_generates_pipeline_id_when_not_passed(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        # Mirrors ResultsManager's local-id pattern (YYYYMMDD_HHMMSS_local).
        # We don't assert the format, only that a run dir was created and its
        # name is reported in the CLI output.
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_enriched_record(in_dir, "src_a", "frase. " * 20)

        result = runner.invoke(app, ["chunk", str(in_dir), "--view", "cep_4k"])
        assert result.exit_code == 0, result.output

        run_dirs = [d for d in results_base.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1, f"expected one run dir, got {run_dirs}"
        run_dir = run_dirs[0]
        assert (run_dir / "chunk" / "outputs" / "cep_4k" / "src_a.json").exists()


class TestAranduChunkCounterSemantics:
    """Counter was ambiguous for multi-view runs — split into source vs artifact counts."""

    def test_multi_view_run_reports_artifact_count_not_source_count(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_enriched_record(in_dir, "src_a", "frase. " * 30)
        _write_enriched_record(in_dir, "src_b", "frase. " * 30)

        result = runner.invoke(
            app,
            [
                "chunk", str(in_dir), "--id", "run_x",
                "--view", "cep_4k", "--view", "nx_2k",
            ],
        )
        assert result.exit_code == 0, result.output

        # CLI must surface the on-disk artifact count, not the source count,
        # so multi-view runs don't undercount.
        out = result.output
        assert "Wrote 4 ChunkSet(s)" in out, out  # 2 sources * 2 views
        assert "2 source(s)" in out
        assert "2 view(s)" in out


class TestAranduChunkResumability:
    """The chunk stage resumes from checkpoint like sibling orchestrators."""

    def test_second_run_with_same_id_skips_completed_sources(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_enriched_record(in_dir, "src_a", "frase. " * 10)
        _write_enriched_record(in_dir, "src_b", "frase. " * 10)

        # First run: chunk both sources.
        first = runner.invoke(
            app, ["chunk", str(in_dir), "--id", "run_x", "--view", "cep_4k"]
        )
        assert first.exit_code == 0, first.output

        outputs = results_base / "run_x" / "chunk" / "outputs" / "cep_4k"
        first_mtime = (outputs / "src_a.json").stat().st_mtime_ns

        # Second run: same --id; checkpoint says both are completed → both skipped.
        second = runner.invoke(
            app, ["chunk", str(in_dir), "--id", "run_x", "--view", "cep_4k"]
        )
        assert second.exit_code == 0, second.output
        assert "Resumed: 2 source(s) already completed" in second.output

        # File was not re-written.
        assert (outputs / "src_a.json").stat().st_mtime_ns == first_mtime


class TestAranduChunkResultsManagerWiring:
    """The CLI emits the standard stage triplet: outputs/, checkpoint, run_metadata."""

    def test_emits_run_metadata_json(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_enriched_record(in_dir, "src_a", "frase. " * 10)

        result = runner.invoke(app, ["chunk", str(in_dir), "--id", "run_x", "--view", "cep_4k"])
        assert result.exit_code == 0, result.output

        meta_path = results_base / "run_x" / "chunk" / "run_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["run_id"] == "run_x"
        assert meta["pipeline_id"] == "run_x"
        assert meta["pipeline_type"] == PipelineType.CHUNK.value
        assert meta["status"] == RunStatus.COMPLETED.value
        assert "config" in meta

    def test_emits_checkpoint_with_completed_file_ids(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_enriched_record(in_dir, "src_a", "frase. " * 10)
        _write_enriched_record(in_dir, "src_b", "frase. " * 10)

        result = runner.invoke(app, ["chunk", str(in_dir), "--id", "run_x", "--view", "cep_4k"])
        assert result.exit_code == 0, result.output

        ckpt_path = results_base / "run_x" / "chunk" / "chunk_checkpoint.json"
        assert ckpt_path.exists()
        ckpt = json.loads(ckpt_path.read_text())
        # CheckpointManager's persisted shape includes a completed set.
        completed = ckpt.get("completed_files") or ckpt.get("completed") or []
        assert set(completed) >= {"src_a", "src_b"}

    def test_emits_pipeline_json_at_run_root(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_enriched_record(in_dir, "src_a", "frase. " * 10)

        result = runner.invoke(app, ["chunk", str(in_dir), "--id", "run_x", "--view", "cep_4k"])
        assert result.exit_code == 0, result.output

        pjson = results_base / "run_x" / "pipeline.json"
        assert pjson.exists()
        payload = json.loads(pjson.read_text())
        assert payload["pipeline_id"] == "run_x"
        assert PipelineType.CHUNK.value in payload["steps_run"]


class TestAranduChunkContentInvariants:
    """Per-view ChunkSet contents are preserved across the path migration."""

    def test_chunk_records_source_text_sha256(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        text = "Esta é uma frase de teste. " * 20
        _write_enriched_record(in_dir, "src_a", text)

        result = runner.invoke(app, ["chunk", str(in_dir), "--id", "run_x", "--view", "cep_4k"])
        assert result.exit_code == 0, result.output

        cs = ChunkSet.load(results_base / "run_x" / "chunk" / "outputs" / "cep_4k" / "src_a.json")
        expected_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert cs.source_text_sha256 == expected_sha

    def test_rejects_unknown_view(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_enriched_record(in_dir, "src_a", "text")

        result = runner.invoke(app, ["chunk", str(in_dir), "--id", "run_x", "--view", "garbage"])
        assert result.exit_code != 0
        assert "Unknown chunker_id" in result.output or "garbage" in result.output

    def test_handles_empty_input_dir(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()

        result = runner.invoke(app, ["chunk", str(in_dir), "--id", "run_x", "--view", "cep_4k"])
        assert result.exit_code == 0
        # No source files → no outputs subdir contents, but the stage scaffolding still lands.
        assert (results_base / "run_x" / "chunk" / "run_metadata.json").exists()
