"""Push creates exactly one project per build, and records it."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from arandu.shared.annotation.build import CONFIG_FILENAME, MANIFEST_FILENAME, TASKS_FILENAME
from arandu.shared.annotation.push import run_push_annotation
from arandu.shared.annotation.schemas import AnnotationManifest

if TYPE_CHECKING:
    from pathlib import Path


class FakeClient:
    """In-memory :class:`LabelStudioClient`."""

    def __init__(self, next_id: int = 42) -> None:
        self.next_id = next_id
        self.created: list[tuple[str, str]] = []
        self.imported: list[tuple[int, list[dict[str, Any]]]] = []

    def create_project(self, title: str, label_config: str) -> int:
        self.created.append((title, label_config))
        project_id = self.next_id
        self.next_id += 1
        return project_id

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> int:
        self.imported.append((project_id, tasks))
        return len(tasks)

    def export_annotations(self, project_id: int) -> list[dict[str, Any]]:
        return []


@pytest.fixture
def built(tmp_path: Path) -> Path:
    outputs = tmp_path / "run-a" / "annotation" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / CONFIG_FILENAME).write_text("<View/>", encoding="utf-8")
    (outputs / TASKS_FILENAME).write_text(
        json.dumps([{"data": {"task_id": 0, "segment": "s", "question": "q", "answer": "a"}}]),
        encoding="utf-8",
    )
    AnnotationManifest(
        pipeline_id="run-a",
        seed=5,
        total_items=1,
        per_cell=15,
        pool_sha256="c" * 64,
        ruler_sha256="d" * 64,
        task_map={"0": "src-a:0"},
    ).save(outputs / MANIFEST_FILENAME)
    return tmp_path


def _manifest(base: Path) -> AnnotationManifest:
    return AnnotationManifest.load(base / "run-a" / "annotation" / "outputs" / MANIFEST_FILENAME)


class TestPush:
    def test_creates_the_project_and_imports_the_tasks(self, built: Path) -> None:
        client = FakeClient()
        project_id = run_push_annotation("run-a", client=client, base_dir=built)
        assert project_id == 42
        assert client.created[0][1] == "<View/>"
        expected_task = {"data": {"task_id": 0, "segment": "s", "question": "q", "answer": "a"}}
        assert client.imported == [(42, [expected_task])]

    def test_records_the_project_id_in_the_manifest(self, built: Path) -> None:
        run_push_annotation("run-a", client=FakeClient(), base_dir=built)
        manifest = _manifest(built)
        assert manifest.project_id == 42
        assert manifest.project_ids == [42]

    def test_default_title_carries_the_run_id(self, built: Path) -> None:
        client = FakeClient()
        run_push_annotation("run-a", client=client, base_dir=built)
        assert "run-a" in client.created[0][0]

    def test_explicit_title_wins(self, built: Path) -> None:
        client = FakeClient()
        run_push_annotation("run-a", client=client, base_dir=built, title="Validade êmica")
        assert client.created[0][0] == "Validade êmica"


class TestDuplicateProtection:
    def test_second_push_refuses_and_names_the_project(self, built: Path) -> None:
        run_push_annotation("run-a", client=FakeClient(), base_dir=built)
        with pytest.raises(ValueError, match="42"):
            run_push_annotation("run-a", client=FakeClient(), base_dir=built)

    def test_second_push_creates_nothing(self, built: Path) -> None:
        run_push_annotation("run-a", client=FakeClient(), base_dir=built)
        second = FakeClient(next_id=99)
        with pytest.raises(ValueError):
            run_push_annotation("run-a", client=second, base_dir=built)
        assert second.created == []

    def test_force_creates_another_and_records_both(self, built: Path) -> None:
        run_push_annotation("run-a", client=FakeClient(), base_dir=built)
        second = run_push_annotation(
            "run-a", client=FakeClient(next_id=99), base_dir=built, force=True
        )
        manifest = _manifest(built)
        assert second == 99
        assert manifest.project_id == 99
        assert manifest.project_ids == [42, 99]


class TestErrors:
    def test_missing_build_names_the_build_command(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="emic-annotation-build"):
            run_push_annotation("absent", client=FakeClient(), base_dir=tmp_path)

    def test_task_count_mismatch_with_the_manifest_fails(self, built: Path) -> None:
        path = built / "run-a" / "annotation" / "outputs" / MANIFEST_FILENAME
        manifest = AnnotationManifest.load(path)
        manifest.total_items = 120
        manifest.save(path)
        with pytest.raises(ValueError, match="120"):
            run_push_annotation("run-a", client=FakeClient(), base_dir=built)
