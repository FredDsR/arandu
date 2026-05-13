"""Tests for `arandu chunk` CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path  # noqa: TC003 — used as a parameter type at runtime

import pytest
from typer.testing import CliRunner

from arandu.cli.app import app
from arandu.shared.chunking.schemas import ChunkSet


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


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


class TestArandruChunkCli:
    def test_chunk_writes_chunkset_per_source(self, runner: CliRunner, tmp_path: Path) -> None:
        in_dir = tmp_path / "results"
        in_dir.mkdir()
        out_dir = tmp_path / "chunks"

        text_a = "Esta é uma frase de teste. " * 50  # ~1350 chars
        _write_enriched_record(in_dir, "src_a", text_a)
        text_b = "Outra frase. " * 30
        _write_enriched_record(in_dir, "src_b", text_b)

        result = runner.invoke(app, ["chunk", str(in_dir), "-o", str(out_dir), "--view", "cep_4k"])
        assert result.exit_code == 0, result.output

        # One ChunkSet per source file
        out_a = out_dir / "src_a.json"
        out_b = out_dir / "src_b.json"
        assert out_a.exists()
        assert out_b.exists()

        cs_a = ChunkSet.load(out_a)
        assert cs_a.source_file_id == "src_a"
        assert "cep_4k" in cs_a.views
        assert len(cs_a.view("cep_4k")) >= 1

    def test_chunk_supports_multiple_views(self, runner: CliRunner, tmp_path: Path) -> None:
        in_dir = tmp_path / "results"
        in_dir.mkdir()
        out_dir = tmp_path / "chunks"

        text = "Esta é uma frase de teste. " * 200  # ~5400 chars
        _write_enriched_record(in_dir, "src_a", text)

        result = runner.invoke(
            app,
            [
                "chunk",
                str(in_dir),
                "-o",
                str(out_dir),
                "--view",
                "cep_4k",
                "--view",
                "nx_2k",
            ],
        )
        assert result.exit_code == 0, result.output

        cs = ChunkSet.load(out_dir / "src_a.json")
        assert set(cs.views) == {"cep_4k", "nx_2k"}

    def test_chunk_records_source_text_sha256(self, runner: CliRunner, tmp_path: Path) -> None:
        in_dir = tmp_path / "results"
        in_dir.mkdir()
        out_dir = tmp_path / "chunks"

        text = "Esta é uma frase de teste. " * 20
        _write_enriched_record(in_dir, "src_a", text)

        result = runner.invoke(app, ["chunk", str(in_dir), "-o", str(out_dir), "--view", "cep_4k"])
        assert result.exit_code == 0, result.output

        cs = ChunkSet.load(out_dir / "src_a.json")
        expected_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert cs.source_text_sha256 == expected_sha

    def test_chunk_rejects_unknown_view(self, runner: CliRunner, tmp_path: Path) -> None:
        in_dir = tmp_path / "results"
        in_dir.mkdir()
        _write_enriched_record(in_dir, "src_a", "text")

        result = runner.invoke(
            app,
            ["chunk", str(in_dir), "-o", str(tmp_path / "chunks"), "--view", "garbage"],
        )
        assert result.exit_code != 0
        assert "Unknown chunker_id" in result.output or "garbage" in result.output

    def test_chunk_handles_empty_input_dir(self, runner: CliRunner, tmp_path: Path) -> None:
        in_dir = tmp_path / "results"
        in_dir.mkdir()
        out_dir = tmp_path / "chunks"

        result = runner.invoke(app, ["chunk", str(in_dir), "-o", str(out_dir), "--view", "cep_4k"])
        # Empty input should complete cleanly with a warning
        assert result.exit_code == 0
