"""Schemas for the annotation instrument, including the blinding contract."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from arandu.shared.annotation.schemas import (
    AnnotationLabel,
    AnnotationManifest,
    AnnotationTask,
)
from arandu.shared.schemas import PipelineType

if TYPE_CHECKING:
    from pathlib import Path


def _manifest(**overrides: object) -> AnnotationManifest:
    base = {
        "pipeline_id": "thesis-run-01",
        "seed": 7,
        "total_items": 2,
        "per_cell": 15,
        "pool_sha256": "a" * 64,
        "ruler_sha256": "b" * 64,
        "task_map": {"0": "src-a:3", "1": "src-b:11"},
    }
    base.update(overrides)
    return AnnotationManifest(**base)  # type: ignore[arg-type]


class TestPipelineType:
    def test_annotation_stage_exists(self) -> None:
        assert PipelineType.ANNOTATION.value == "annotation"


class TestAnnotationTask:
    def test_serializes_exactly_the_blinded_fields(self) -> None:
        task = AnnotationTask(task_id=0, segment="s", question="q", answer="a")
        assert set(task.model_dump().keys()) == {"task_id", "segment", "question", "answer"}

    def test_rejects_unknown_fields(self) -> None:
        """Extra keys are forbidden: a leak must be a crash, not a silent field."""
        with pytest.raises(ValidationError):
            AnnotationTask(task_id=0, segment="s", question="q", answer="a", bloom_level="analyze")

    def test_to_label_studio_wraps_in_data(self) -> None:
        task = AnnotationTask(task_id=4, segment="s", question="q", answer="a")
        assert task.to_label_studio() == {
            "data": {"task_id": 4, "segment": "s", "question": "q", "answer": "a"}
        }


class TestAnnotationManifest:
    def test_round_trips_through_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        _manifest().save(path)
        assert AnnotationManifest.load(path) == _manifest()

    def test_project_id_defaults_to_unpushed(self) -> None:
        manifest = _manifest()
        assert manifest.project_id is None
        assert manifest.project_ids == []

    def test_pair_id_for_resolves_the_join(self) -> None:
        assert _manifest().pair_id_for(1) == "src-b:11"

    def test_pair_id_for_unknown_task_raises(self) -> None:
        with pytest.raises(KeyError):
            _manifest().pair_id_for(99)

    def test_task_map_is_never_written_to_label_studio_shape(self, tmp_path: Path) -> None:
        """The join lives here and only here."""
        path = tmp_path / "manifest.json"
        _manifest().save(path)
        assert "src-a:3" in json.loads(path.read_text(encoding="utf-8"))["task_map"].values()


class TestAnnotationLabel:
    def test_score_is_constrained_to_the_ordinal_range(self) -> None:
        with pytest.raises(ValidationError):
            AnnotationLabel(
                pair_id="a:0", annotator_id="A1", score=6, rationale=None, timestamp="t"
            )

    def test_rationale_is_optional(self) -> None:
        label = AnnotationLabel(
            pair_id="a:0", annotator_id="A1", score=3, rationale=None, timestamp="t"
        )
        assert label.rationale is None
