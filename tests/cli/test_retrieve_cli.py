"""Tests for ``arandu retrieve`` CLI command.

Locks the surface arrived at by the design proposal in
``retrieve-cli-design.md``: ``--id`` required, ``--arm`` repeatable
with a sane default, clear failure modes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from arandu.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def results_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "results"
    monkeypatch.setenv("ARANDU_RESULTS_BASE_DIR", str(base))
    return base


def _seed_cep(base: Path, pipeline_id: str = "run_x") -> None:
    cep_dir = base / pipeline_id / "cep" / "outputs"
    cep_dir.mkdir(parents=True)
    record = {
        "source_file_id": "src_a",
        "source_filename": "src_a.mp4",
        "transcription_text": "Texto.",
        "chunker_id": "cep_4k",
        "qa_pairs": [
            {
                "question": "Q1?",
                "answer": "A1.",
                "context": "C1.",
                "question_type": "factual",
                "confidence": 0.9,
                "bloom_level": "remember",
                "chunk_id": "chk_00",
            }
        ],
        "model_id": "qwen3:14b",
        "provider": "ollama",
        "language": "pt",
        "total_pairs": 1,
        "bloom_distribution": {},
    }
    (cep_dir / "src_a_cep_qa.json").write_text(json.dumps(record))


class TestRetrieveCli:
    def test_null_arm_end_to_end(self, runner: CliRunner, results_base: Path) -> None:
        _seed_cep(results_base)
        result = runner.invoke(app, ["retrieve", "--id", "run_x", "--arm", "null", "--top-k", "5"])

        assert result.exit_code == 0, result.output
        outputs = results_base / "run_x" / "retrieve" / "outputs" / "null" / "cep"
        assert len(list(outputs.glob("*.json"))) == 1
        assert "Wrote 1 RetrievalRecord" in result.output

    def test_default_arms_includes_four(self, runner: CliRunner, results_base: Path) -> None:
        # With no --arm, the CLI runs the 4 non-atlas_rag arms. Three of
        # them (bm25 / khop_passage / khop_triple) fail at construction
        # because no chunks/KG are seeded; the null arm completes.
        _seed_cep(results_base)
        result = runner.invoke(app, ["retrieve", "--id", "run_x"])

        assert result.exit_code == 0, result.output
        # null arm should have emitted at least one record.
        null_outputs = results_base / "run_x" / "retrieve" / "outputs" / "null" / "cep"
        assert null_outputs.exists()
        assert "Failed retrievals" in result.output

    def test_missing_cep_fails_with_clear_error(
        self, runner: CliRunner, results_base: Path
    ) -> None:
        result = runner.invoke(app, ["retrieve", "--id", "never_built", "--arm", "null"])
        assert result.exit_code == 1
        assert "CEP outputs not found" in result.output

    def test_top_k_zero_rejected_by_typer(self, runner: CliRunner, results_base: Path) -> None:
        # Typer's min=1 catches this before any business logic runs.
        _seed_cep(results_base)
        result = runner.invoke(app, ["retrieve", "--id", "run_x", "--arm", "null", "--top-k", "0"])
        assert result.exit_code != 0

    def test_help_renders(self, runner: CliRunner) -> None:
        # Rich/Typer's help panel wraps + truncates inside CliRunner's
        # default terminal width, making content-level assertions brittle.
        # Lock the structural surface only: command is registered, --help
        # exits 0, and the synopsis line is present.
        result = runner.invoke(app, ["retrieve", "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "retrieve" in result.output
