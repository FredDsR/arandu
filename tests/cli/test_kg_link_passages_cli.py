"""Tests for ``arandu kg-link-passages`` CLI command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from arandu.cli.app import app
from arandu.kg.passage_offsets import PassageOffsetSidecar

if TYPE_CHECKING:
    from pathlib import Path


_HEADER = "[Contexto da Entrevista]\nLocal: BARRA DE PELOTAS\n[Transcrição]\n"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def results_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "results"
    monkeypatch.setenv("ARANDU_RESULTS_BASE_DIR", str(base))
    return base


def _seed_fixture(base: Path, pipeline_id: str = "run_x") -> str:
    tr_out = base / pipeline_id / "transcription" / "outputs"
    tr_out.mkdir(parents=True)
    text = "Esta é uma transcrição de teste sobre enchentes."
    (tr_out / "src_a.json").write_text(
        json.dumps(
            {
                "gdrive_id": "src_a",
                "name": "src_a.mp4",
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
        )
    )

    kg_ext = base / pipeline_id / "kg" / "outputs" / "atlas_output" / "kg_extraction"
    kg_ext.mkdir(parents=True)
    (kg_ext / "qwen3_extraction.json").write_text(
        json.dumps({"id": "src_a", "original_text": _HEADER + text, "metadata": {"lang": "pt"}})
        + "\n"
    )
    return pipeline_id


class TestKgLinkPassagesCli:
    def test_writes_sidecar_to_default_kg_outputs_path(
        self, runner: CliRunner, results_base: Path
    ) -> None:
        pid = _seed_fixture(results_base)

        result = runner.invoke(app, ["kg-link-passages", "--id", pid])
        assert result.exit_code == 0, result.output

        sidecar_path = results_base / pid / "kg" / "outputs" / "passage_offsets.json"
        assert sidecar_path.exists()
        sidecar = PassageOffsetSidecar.load(sidecar_path)
        assert sidecar.kg_run_id == pid
        assert len(sidecar.offsets) == 1
        assert sidecar.offsets[0].source_file_id == "src_a"

    def test_output_flag_overrides_default_path(
        self, runner: CliRunner, tmp_path: Path, results_base: Path
    ) -> None:
        pid = _seed_fixture(results_base)
        custom_out = tmp_path / "custom" / "offsets.json"

        result = runner.invoke(app, ["kg-link-passages", "--id", pid, "--output", str(custom_out)])
        assert result.exit_code == 0, result.output
        assert custom_out.exists()
        # Default location must NOT be written when --output is supplied.
        default_path = results_base / pid / "kg" / "outputs" / "passage_offsets.json"
        assert not default_path.exists()

    def test_missing_pipeline_id_exits_with_error(
        self, runner: CliRunner, results_base: Path
    ) -> None:
        result = runner.invoke(app, ["kg-link-passages", "--id", "no_such_run"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "no_such_run" in result.output

    def test_reports_unmatched_count_when_passages_cannot_anchor(
        self, runner: CliRunner, results_base: Path
    ) -> None:
        pid = "run_with_orphan"
        # Build a fixture where the chunk text is not in the source transcription.
        tr_out = results_base / pid / "transcription" / "outputs"
        tr_out.mkdir(parents=True)
        (tr_out / "src_a.json").write_text(
            json.dumps(
                {
                    "gdrive_id": "src_a",
                    "name": "src_a.mp4",
                    "mimeType": "video/mp4",
                    "parents": ["folder"],
                    "webContentLink": "https://drive.google.com/test",
                    "size_bytes": 1024,
                    "duration_milliseconds": 60000,
                    "transcription_text": "Texto fonte legítimo.",
                    "detected_language": "pt",
                    "language_probability": 0.95,
                    "model_id": "whisper-large-v3",
                    "compute_device": "cpu",
                    "processing_duration_sec": 30.5,
                    "transcription_status": "completed",
                }
            )
        )
        kg_ext = results_base / pid / "kg" / "outputs" / "atlas_output" / "kg_extraction"
        kg_ext.mkdir(parents=True)
        (kg_ext / "ext.json").write_text(
            json.dumps(
                {
                    "id": "src_a",
                    "original_text": _HEADER + "Texto que não existe na fonte.",
                    "metadata": {"lang": "pt"},
                }
            )
            + "\n"
        )

        result = runner.invoke(app, ["kg-link-passages", "--id", pid])
        assert result.exit_code == 0, result.output
        assert "unmatched" in result.output.lower()
