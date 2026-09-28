"""Tests for the judge-qa run-metadata snapshot helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arandu.qa.cep.judge_run import (
    GATE_CRITERIA,
    resolve_pipeline_layout,
    snapshot_criteria,
)

if TYPE_CHECKING:
    from pathlib import Path


def _criteria_dir(tmp_path: Path) -> Path:
    root = tmp_path / "criteria"
    for name in GATE_CRITERIA:
        (root / name / "pt").mkdir(parents=True)
        (root / name / "config.json").write_text('{"threshold": 0.625}')
        (root / name / "pt" / "prompt.md").write_text(f"prompt {name}")
    return root


def test_resolves_results_layout(tmp_path: Path) -> None:
    outputs = tmp_path / "results" / "run-x" / "cep" / "outputs"
    outputs.mkdir(parents=True)

    assert resolve_pipeline_layout(outputs) == ((tmp_path / "results").resolve(), "run-x")


def test_rejects_non_results_layout(tmp_path: Path) -> None:
    assert resolve_pipeline_layout(tmp_path) is None
    other = tmp_path / "run-x" / "kg" / "outputs"
    other.mkdir(parents=True)
    assert resolve_pipeline_layout(other) is None


def test_snapshot_reads_thresholds(tmp_path: Path) -> None:
    snapshot = snapshot_criteria(_criteria_dir(tmp_path), "pt")

    assert set(snapshot) == set(GATE_CRITERIA)
    assert all(c.threshold == 0.625 for c in snapshot.values())


def test_prompt_edit_changes_only_its_digest(tmp_path: Path) -> None:
    """Two runs are replicas only if their digests match, so an edit must show."""
    root = _criteria_dir(tmp_path)
    before = snapshot_criteria(root, "pt")

    (root / "informativeness" / "pt" / "prompt.md").write_text("edited")
    after = snapshot_criteria(root, "pt")

    assert after["informativeness"].sha256 != before["informativeness"].sha256
    assert after["faithfulness"].sha256 == before["faithfulness"].sha256


def test_missing_config_yields_no_threshold(tmp_path: Path) -> None:
    root = _criteria_dir(tmp_path)
    (root / "faithfulness" / "config.json").unlink()

    assert snapshot_criteria(root, "pt")["faithfulness"].threshold is None
