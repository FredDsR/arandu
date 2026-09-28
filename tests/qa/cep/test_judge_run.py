"""Tests for the judge-qa run-metadata snapshot helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from arandu.qa.cep.judge_run import (
    GATE_CRITERIA,
    archive_previous_snapshot,
    has_judge_error,
    resolve_pipeline_layout,
    snapshot_criteria,
)
from arandu.shared.judge.schemas import CriterionScore, JudgePipelineResult, JudgeStepResult

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
    (tmp_path / "results" / "run-x" / "pipeline.json").write_text("{}")

    assert resolve_pipeline_layout(outputs) == ((tmp_path / "results").resolve(), "run-x")


def test_path_shape_alone_is_not_a_results_layout(tmp_path: Path) -> None:
    """``~/datasets/cep/outputs`` has the shape but no pipeline.json."""
    outputs = tmp_path / "datasets" / "cep" / "outputs"
    outputs.mkdir(parents=True)

    assert resolve_pipeline_layout(outputs) is None


def test_rejects_non_results_layout(tmp_path: Path) -> None:
    assert resolve_pipeline_layout(tmp_path) is None
    other = tmp_path / "run-x" / "kg" / "outputs"
    other.mkdir(parents=True)
    assert resolve_pipeline_layout(other) is None


def test_snapshot_reads_thresholds(tmp_path: Path) -> None:
    snapshot = snapshot_criteria(_criteria_dir(tmp_path), "pt", 0.1)

    assert set(snapshot) == set(GATE_CRITERIA)
    assert all(c.threshold == 0.625 for c in snapshot.values())


def test_prompt_edit_changes_only_its_digest(tmp_path: Path) -> None:
    """Two runs are replicas only if their digests match, so an edit must show."""
    root = _criteria_dir(tmp_path)
    before = snapshot_criteria(root, "pt", 0.1)

    (root / "informativeness" / "pt" / "prompt.md").write_text("edited")
    after = snapshot_criteria(root, "pt", 0.1)

    assert after["informativeness"].sha256 != before["informativeness"].sha256
    assert after["faithfulness"].sha256 == before["faithfulness"].sha256


def test_missing_config_yields_no_threshold_and_no_digest(tmp_path: Path) -> None:
    root = _criteria_dir(tmp_path)
    (root / "faithfulness" / "config.json").unlink()

    snapshot = snapshot_criteria(root, "pt", 0.1)["faithfulness"]
    assert snapshot.threshold is None
    assert snapshot.sha256 is None


def test_missing_prompt_yields_no_digest(tmp_path: Path) -> None:
    """Two runs without prompts must not compare as byte-identical."""
    root = _criteria_dir(tmp_path)

    assert all(c.sha256 is None for c in snapshot_criteria(root, "en", 0.1).values())


def test_criterion_temperature_override_is_recorded(tmp_path: Path) -> None:
    root = _criteria_dir(tmp_path)
    (root / "informativeness" / "config.json").write_text(
        '{"threshold": 0.625, "temperature": 0.7}'
    )

    snapshot = snapshot_criteria(root, "pt", 0.1)
    assert snapshot["informativeness"].temperature == 0.7
    assert snapshot["faithfulness"].temperature == 0.1


def test_archive_previous_snapshot_keeps_history(tmp_path: Path) -> None:
    step = tmp_path / "judge_qa"
    step.mkdir()
    assert archive_previous_snapshot(step) is None

    for started in ("2026-09-28T10:00:00Z", "2026-09-28T10:00:00Z"):
        (step / "run_metadata.json").write_text(json.dumps({"started_at": started}))
        archived = archive_previous_snapshot(step)
        assert archived is not None and archived.parent == step / "history"

    assert not (step / "run_metadata.json").exists()
    assert len(list((step / "history").glob("run_metadata.*.json"))) == 2


def _pipeline(*errors: str | None) -> JudgePipelineResult:
    scores = {
        f"c{i}": CriterionScore(score=1.0, threshold=0.625, rationale="r", error=e)
        for i, e in enumerate(errors)
    }
    step = JudgeStepResult(criterion_scores=scores)
    return JudgePipelineResult(stage_results={"s": step}, passed=step.passed)


def test_has_judge_error_detects_any_criterion_error() -> None:
    assert has_judge_error(_pipeline(None, "timeout"))
    assert not has_judge_error(_pipeline(None, None))
