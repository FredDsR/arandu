"""Tests for the `arandu judge-qa` CLI command.

The command's contract is grounding symmetry: every pair must be judged
against the chunk it was generated from (``QAPairCEP.context``), never
against the whole transcription.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from arandu.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class _RecordingJudge:
    """Judge double that records the context each pair was judged against."""

    def __init__(self, **_: Any) -> None:
        self.calls: list[tuple[str, str]] = []

    def validate(self, qa_pair: Any, context: str) -> Any:
        self.calls.append((qa_pair.question, context))
        return qa_pair


def _write_record(dir_: Path, *, metadata: dict[str, str] | None = None) -> Path:
    """Write a two-chunk CEP QA record whose pairs carry distinct contexts."""
    payload: dict[str, Any] = {
        "source_gdrive_id": "file-1",
        "source_filename": "entrevista.mp4",
        "source_metadata_context_enabled": metadata is not None,
        "transcription_text": "CHUNK_A texto inicial. CHUNK_B texto final.",
        "model_id": "test-model",
        "provider": "custom",
        "language": "pt",
        "total_pairs": 2,
        "qa_pairs": [
            {
                "question": "Q1",
                "answer": "A1",
                "context": "CHUNK_A texto inicial.",
                "question_type": "factual",
                "bloom_level": "remember",
            },
            {
                "question": "Q2",
                "answer": "A2",
                "context": "CHUNK_B texto final.",
                "question_type": "conceptual",
                "bloom_level": "analyze",
            },
        ],
    }
    if metadata is not None:
        payload["source_metadata"] = metadata

    path = dir_ / "file-1_cep_qa.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> _RecordingJudge:
    """Patch the judge and its LLM client so the command runs offline."""
    import arandu.qa.cep.judge as judge_module
    import arandu.transcription.judge as validator_module

    recording = _RecordingJudge()
    monkeypatch.setattr(judge_module, "QAJudge", lambda **kwargs: recording)
    monkeypatch.setattr(validator_module, "build_validator_client", lambda **kwargs: object())
    return recording


def test_judges_each_pair_against_its_own_chunk(
    tmp_path: Path, runner: CliRunner, judge: _RecordingJudge
) -> None:
    """Each pair is grounded on its originating chunk, not the transcription."""
    _write_record(tmp_path)

    result = runner.invoke(app, ["judge-qa", str(tmp_path), "--model", "test-model"])

    assert result.exit_code == 0, result.output
    assert judge.calls == [
        ("Q1", "CHUNK_A texto inicial."),
        ("Q2", "CHUNK_B texto final."),
    ]


def test_appends_metadata_to_the_chunk_context(
    tmp_path: Path, runner: CliRunner, judge: _RecordingJudge
) -> None:
    """Metadata symmetry survives the switch to chunk-scoped grounding."""
    _write_record(tmp_path, metadata={"location": "DOQUINHAS"})

    result = runner.invoke(app, ["judge-qa", str(tmp_path), "--model", "test-model"])

    assert result.exit_code == 0, result.output
    metadata_block = "\n\nMetadados da Entrevista:\n- Local: DOQUINHAS"
    assert judge.calls == [
        ("Q1", f"CHUNK_A texto inicial.{metadata_block}"),
        ("Q2", f"CHUNK_B texto final.{metadata_block}"),
    ]


def test_falls_back_to_the_transcription_for_contextless_pairs(
    tmp_path: Path, runner: CliRunner, judge: _RecordingJudge
) -> None:
    """A legacy pair with an empty context is still judged, against the document."""
    path = _write_record(tmp_path)
    payload = json.loads(path.read_text())
    payload["qa_pairs"][1]["context"] = ""
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(app, ["judge-qa", str(tmp_path), "--model", "test-model"])

    assert result.exit_code == 0, result.output
    assert judge.calls[1] == ("Q2", "CHUNK_A texto inicial. CHUNK_B texto final.")


def _results_layout(tmp_path: Path, pipeline_id: str = "run-x") -> Path:
    """Create ``<base>/<id>/cep/outputs`` and return it."""
    outputs = tmp_path / "results" / pipeline_id / "cep" / "outputs"
    outputs.mkdir(parents=True)
    return outputs


def test_writes_run_metadata_for_a_results_layout(
    tmp_path: Path,
    runner: CliRunner,
    judge: _RecordingJudge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run over results/<id>/cep/outputs records how its verdicts were produced."""
    monkeypatch.delenv("ARANDU_JUDGE_TEMPERATURE", raising=False)
    outputs = _results_layout(tmp_path)
    _write_record(outputs)

    result = runner.invoke(app, ["judge-qa", str(outputs), "--model", "test-model"])

    assert result.exit_code == 0, result.output
    pipeline_dir = tmp_path / "results" / "run-x"
    metadata = json.loads((pipeline_dir / "judge_qa" / "run_metadata.json").read_text())
    assert metadata["pipeline_type"] == "judge_qa"
    assert metadata["status"] == "completed"
    assert metadata["total_items"] == 2
    values = metadata["config"]["config_values"]
    assert values["mode"] == "resume"
    assert values["model_id"] == "test-model"
    assert values["judge"]["temperature"] == 0.1
    assert set(values["criteria"]) == {
        "faithfulness",
        "bloom_calibration",
        "informativeness",
        "self_containedness",
    }
    assert all(c["threshold"] == 0.625 for c in values["criteria"].values())
    assert values["bloom_descriptions_sha256"]
    steps = json.loads((pipeline_dir / "pipeline.json").read_text())["steps_run"]
    assert "judge_qa" in steps


def test_records_rejudge_mode(tmp_path: Path, runner: CliRunner, judge: _RecordingJudge) -> None:
    """The mode is a CLI flag, so it must reach the snapshot explicitly."""
    outputs = _results_layout(tmp_path)
    _write_record(outputs)

    result = runner.invoke(app, ["judge-qa", str(outputs), "--model", "test-model", "--rejudge"])

    assert result.exit_code == 0, result.output
    metadata_path = tmp_path / "results" / "run-x" / "judge_qa" / "run_metadata.json"
    assert json.loads(metadata_path.read_text())["config"]["config_values"]["mode"] == "rejudge"


def test_skips_run_metadata_outside_the_results_layout(
    tmp_path: Path, runner: CliRunner, judge: _RecordingJudge
) -> None:
    """An ad hoc dataset directory is judged without creating a step directory."""
    _write_record(tmp_path)

    result = runner.invoke(app, ["judge-qa", str(tmp_path), "--model", "test-model"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path.parent / "judge_qa").exists()
    assert list(tmp_path.glob("**/run_metadata.json")) == []
