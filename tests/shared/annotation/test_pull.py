"""Pull anonymizes annotators and refuses to guess on a desynced project."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from arandu.shared.annotation.build import MANIFEST_FILENAME
from arandu.shared.annotation.pull import (
    ANNOTATOR_MAP_FILENAME,
    LABELS_DIRNAME,
    anonymize,
    run_pull_annotation,
)
from arandu.shared.annotation.schemas import AnnotationLabel, AnnotationManifest

if TYPE_CHECKING:
    from pathlib import Path


def _annotation(user_id: int, choice: str, rationale: str | None = None) -> dict[str, Any]:
    result: list[dict[str, Any]] = [
        {"from_name": "score", "type": "choices", "value": {"choices": [choice]}}
    ]
    if rationale is not None:
        result.append(
            {"from_name": "rationale", "type": "textarea", "value": {"text": [rationale]}}
        )
    return {
        "completed_by": user_id,
        "created_at": "2026-09-01T10:00:00Z",
        "result": result,
    }


def _task(task_id: int, annotations: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": 100 + task_id, "data": {"task_id": task_id}, "annotations": annotations}


class FakeClient:
    def __init__(self, export: list[dict[str, Any]]) -> None:
        self.export = export
        self.exported: list[int] = []

    def create_project(  # pragma: no cover
        self, title: str, label_config: str, *, settings: dict[str, Any] | None = None
    ) -> int:
        raise AssertionError("pull must not create projects")

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> int:  # pragma: no cover
        raise AssertionError("pull must not import tasks")

    def export_annotations(self, project_id: int) -> list[dict[str, Any]]:
        self.exported.append(project_id)
        return self.export


@pytest.fixture
def pushed(tmp_path: Path) -> Path:
    outputs = tmp_path / "run-a" / "annotation" / "outputs"
    outputs.mkdir(parents=True)
    AnnotationManifest(
        pipeline_id="run-a",
        seed=5,
        total_items=3,
        per_cell=15,
        pool_sha256="c" * 64,
        ruler_sha256="d" * 64,
        task_map={"0": "src-a:0", "1": "src-a:4", "2": "src-b:2"},
        project_id=42,
        project_ids=[42],
    ).save(outputs / MANIFEST_FILENAME)
    return tmp_path


def _labels(base: Path, annotator: str) -> list[AnnotationLabel]:
    path = base / "run-a" / "annotation" / "outputs" / LABELS_DIRNAME / f"{annotator}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [AnnotationLabel.model_validate_json(line) for line in fh if line.strip()]


def _cancelled(user_id: int) -> dict[str, Any]:
    """Label Studio's Skip button: a real annotation with no rating in it."""
    return {
        "completed_by": user_id,
        "created_at": "2026-09-01T10:00:00Z",
        "was_cancelled": True,
        "result": [],
    }


def _annotator_map(base: Path) -> dict[str, str]:
    path = base / "run-a" / "annotation" / "outputs" / ANNOTATOR_MAP_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


class TestAnonymize:
    def test_assigns_a_prefixed_id_per_user_in_sorted_order(self) -> None:
        assert anonymize([77, 3, 51]) == {3: "A1", 51: "A2", 77: "A3"}

    def test_is_stable_across_calls(self) -> None:
        assert anonymize([77, 3]) == anonymize([3, 77])

    def test_existing_bindings_are_preserved_verbatim(self) -> None:
        assert anonymize([5, 12, 30], {12: "A1", 30: "A2"}) == {12: "A1", 30: "A2", 5: "A3"}

    def test_new_ids_continue_the_numbering(self) -> None:
        assert anonymize([1, 2], {9: "A1"}) == {9: "A1", 1: "A2", 2: "A3"}


class TestPull:
    def test_writes_one_jsonl_per_annotator(self, pushed: Path) -> None:
        export = [
            _task(0, [_annotation(7, "5 - Preserva."), _annotation(9, "4 - Apaga.")]),
            _task(1, [_annotation(7, "3 - Acrescenta.")]),
        ]
        run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        assert len(_labels(pushed, "A1")) == 2
        assert len(_labels(pushed, "A2")) == 1

    def test_joins_task_id_back_to_pair_id(self, pushed: Path) -> None:
        export = [_task(1, [_annotation(7, "3 - Acrescenta.")])]
        run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        assert _labels(pushed, "A1")[0].pair_id == "src-a:4"

    def test_parses_the_leading_integer_of_the_choice_label(self, pushed: Path) -> None:
        export = [_task(0, [_annotation(7, "4 - Preserva o sentido, mas apaga.")])]
        run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        assert _labels(pushed, "A1")[0].score == 4

    def test_optional_rationale_round_trips(self, pushed: Path) -> None:
        export = [_task(0, [_annotation(7, "2 - Troca o motivo.", rationale="trocou o motivo")])]
        run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        assert _labels(pushed, "A1")[0].rationale == "trocou o motivo"

    def test_absent_rationale_is_none(self, pushed: Path) -> None:
        export = [_task(0, [_annotation(7, "5 - Preserva.")])]
        run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        assert _labels(pushed, "A1")[0].rationale is None

    def test_partial_annotation_is_accepted_and_counted(self, pushed: Path) -> None:
        export = [_task(0, [_annotation(7, "5 - Preserva.")]), _task(1, []), _task(2, [])]
        summary = run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        assert summary.annotators == {"A1": 1}
        assert summary.total_items == 3

    def test_uses_the_project_id_from_the_manifest(self, pushed: Path) -> None:
        client = FakeClient([])
        run_pull_annotation("run-a", client=client, base_dir=pushed)
        assert client.exported == [42]


class TestAnonymity:
    def test_annotator_map_is_written_outside_the_labels_dir(self, pushed: Path) -> None:
        export = [_task(0, [_annotation(7, "5 - Preserva.")])]
        run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        outputs = pushed / "run-a" / "annotation" / "outputs"
        assert (outputs / ANNOTATOR_MAP_FILENAME).exists()
        assert not (outputs / LABELS_DIRNAME / ANNOTATOR_MAP_FILENAME).exists()

    def test_no_email_or_user_id_appears_in_any_label_file(self, pushed: Path) -> None:
        export = [
            _task(
                0,
                [
                    {
                        **_annotation(7, "5 - Preserva."),
                        "completed_by": {"id": 7, "email": "alguem@exemplo.org"},
                    }
                ],
            )
        ]
        run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        raw = (pushed / "run-a" / "annotation" / "outputs" / LABELS_DIRNAME / "A1.jsonl").read_text(
            encoding="utf-8"
        )
        assert "alguem@exemplo.org" not in raw
        assert "completed_by" not in raw


class TestSkippedAnnotations:
    """A skip is a task nobody rated, not a broken export."""

    def test_a_cancelled_annotation_does_not_abort_the_pull(self, pushed: Path) -> None:
        export = [
            _task(0, [_annotation(7, "5 - Preserva."), _cancelled(9)]),
            _task(1, [_annotation(9, "4 - Apaga.")]),
        ]
        summary = run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        assert summary.annotators == {"A1": 1, "A2": 1}

    def test_skips_are_counted_in_the_summary(self, pushed: Path) -> None:
        export = [_task(0, [_cancelled(9)]), _task(1, [_cancelled(9)])]
        summary = run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        assert summary.skipped == 2
        assert summary.annotators == {}

    def test_a_skip_writes_no_label(self, pushed: Path) -> None:
        export = [_task(0, [_cancelled(7)]), _task(1, [_annotation(7, "3 - Acrescenta.")])]
        run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        assert [label.pair_id for label in _labels(pushed, "A1")] == ["src-a:4"]

    def test_an_empty_result_without_the_cancel_flag_is_also_a_skip(self, pushed: Path) -> None:
        export = [
            _task(0, [{"completed_by": 7, "created_at": "2026-09-01T10:00:00Z", "result": []}])
        ]
        summary = run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        assert summary.skipped == 1

    def test_a_present_but_empty_score_region_still_raises(self, pushed: Path) -> None:
        """A malformed rating is the measurement going missing; it must not pass."""
        export = [
            _task(
                0,
                [
                    {
                        "completed_by": 7,
                        "created_at": "2026-09-01T10:00:00Z",
                        "result": [
                            {"from_name": "score", "type": "choices", "value": {"choices": []}}
                        ],
                    }
                ],
            )
        ]
        with pytest.raises(ValueError, match="carries no choice"):
            run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)


class TestAliasStability:
    """An alias, once written, belongs to that user id forever."""

    def test_a_second_pull_with_a_new_annotator_preserves_the_first_aliases(
        self, pushed: Path
    ) -> None:
        first = [_task(0, [_annotation(12, "5 - Preserva."), _annotation(30, "4 - Apaga.")])]
        run_pull_annotation("run-a", client=FakeClient(first), base_dir=pushed)
        assert _annotator_map(pushed) == {"12": "A1", "30": "A2"}

        second = [
            _task(
                0,
                [
                    _annotation(12, "5 - Preserva."),
                    _annotation(30, "4 - Apaga."),
                    _annotation(5, "3 - Acrescenta."),
                ],
            )
        ]
        run_pull_annotation("run-a", client=FakeClient(second), base_dir=pushed)
        assert _annotator_map(pushed) == {"5": "A3", "12": "A1", "30": "A2"}

    def test_a_1_keeps_the_same_persons_labels_across_pulls(self, pushed: Path) -> None:
        first = [_task(0, [_annotation(12, "5 - Preserva.")])]
        run_pull_annotation("run-a", client=FakeClient(first), base_dir=pushed)
        assert _labels(pushed, "A1")[0].score == 5

        second = [
            _task(0, [_annotation(12, "5 - Preserva.")]),
            _task(1, [_annotation(5, "1 - Troca.")]),
        ]
        run_pull_annotation("run-a", client=FakeClient(second), base_dir=pushed)
        assert _labels(pushed, "A1")[0].score == 5
        assert _labels(pushed, "A2")[0].score == 1


class TestAmbiguousProject:
    """A --force re-push leaves two live projects and no safe default."""

    @pytest.fixture
    def two_projects(self, pushed: Path) -> Path:
        path = pushed / "run-a" / "annotation" / "outputs" / MANIFEST_FILENAME
        manifest = AnnotationManifest.load(path)
        manifest.project_id = 99
        manifest.project_ids = [42, 99]
        manifest.save(path)
        return pushed

    def test_several_recorded_projects_refuse_to_pull(self, two_projects: Path) -> None:
        client = FakeClient([])
        with pytest.raises(ValueError, match="--project-id"):
            run_pull_annotation("run-a", client=client, base_dir=two_projects)
        assert client.exported == []

    def test_the_error_lists_every_recorded_id(self, two_projects: Path) -> None:
        with pytest.raises(ValueError, match="42, 99"):
            run_pull_annotation("run-a", client=FakeClient([]), base_dir=two_projects)

    def test_an_explicit_project_id_resolves_the_ambiguity(self, two_projects: Path) -> None:
        client = FakeClient([_task(0, [_annotation(7, "5 - Preserva.")])])
        summary = run_pull_annotation("run-a", client=client, base_dir=two_projects, project_id=42)
        assert client.exported == [42]
        assert summary.project_id == 42

    def test_a_file_export_is_never_ambiguous(self, two_projects: Path, tmp_path: Path) -> None:
        export_path = tmp_path / "export.json"
        export_path.write_text(
            json.dumps([_task(0, [_annotation(7, "5 - Preserva.")])]), encoding="utf-8"
        )
        summary = run_pull_annotation("run-a", export_file=export_path, base_dir=two_projects)
        assert summary.annotators == {"A1": 1}


class TestDuplicateRatings:
    def test_the_same_annotator_rating_a_pair_twice_raises(self, pushed: Path) -> None:
        export = [_task(0, [_annotation(7, "5 - Preserva."), _annotation(7, "2 - Troca.")])]
        with pytest.raises(ValueError, match="src-a:0"):
            run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)

    def test_the_error_names_the_annotator(self, pushed: Path) -> None:
        export = [_task(0, [_annotation(7, "5 - Preserva."), _annotation(7, "2 - Troca.")])]
        with pytest.raises(ValueError, match="A1"):
            run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)

    def test_two_annotators_on_the_same_pair_are_fine(self, pushed: Path) -> None:
        export = [_task(0, [_annotation(7, "5 - Preserva."), _annotation(9, "2 - Troca.")])]
        summary = run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)
        assert summary.annotators == {"A1": 1, "A2": 1}


class TestDesyncAndErrors:
    def test_json_min_export_names_the_export_type(self, pushed: Path) -> None:
        """JSON_MIN drops `data`, so the join key is simply absent."""
        export = [{"id": 100, "score": "5 - Preserva.", "annotations": []}]
        with pytest.raises(ValueError, match="JSON_MIN"):
            run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)

    def test_unknown_task_id_is_a_hard_failure(self, pushed: Path) -> None:
        export = [_task(99, [_annotation(7, "5 - Preserva.")])]
        with pytest.raises(KeyError, match="99"):
            run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)

    def test_unparseable_score_fails_rather_than_guessing(self, pushed: Path) -> None:
        export = [_task(0, [_annotation(7, "muito bom")])]
        with pytest.raises(ValueError, match="score"):
            run_pull_annotation("run-a", client=FakeClient(export), base_dir=pushed)

    def test_unpushed_run_names_the_push_command(self, tmp_path: Path) -> None:
        outputs = tmp_path / "run-a" / "annotation" / "outputs"
        outputs.mkdir(parents=True)
        AnnotationManifest(
            pipeline_id="run-a",
            seed=5,
            total_items=1,
            per_cell=15,
            pool_sha256="c" * 64,
            ruler_sha256="d" * 64,
            task_map={"0": "src-a:0"},
        ).save(outputs / MANIFEST_FILENAME)
        with pytest.raises(ValueError, match="emic-annotation-push"):
            run_pull_annotation("run-a", client=FakeClient([]), base_dir=tmp_path)

    def test_missing_build_names_the_build_command(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="emic-annotation-build"):
            run_pull_annotation("absent", client=FakeClient([]), base_dir=tmp_path)


class TestFileExport:
    def test_reads_a_downloaded_export_without_touching_the_network(
        self, pushed: Path, tmp_path: Path
    ) -> None:
        export_path = tmp_path / "export.json"
        export_path.write_text(
            json.dumps([_task(0, [_annotation(7, "5 - Preserva.")])]), encoding="utf-8"
        )
        summary = run_pull_annotation("run-a", export_file=export_path, base_dir=pushed)
        assert summary.annotators == {"A1": 1}

    def test_file_export_works_on_an_unpushed_run(self, tmp_path: Path) -> None:
        outputs = tmp_path / "run-a" / "annotation" / "outputs"
        outputs.mkdir(parents=True)
        AnnotationManifest(
            pipeline_id="run-a",
            seed=5,
            total_items=1,
            per_cell=15,
            pool_sha256="c" * 64,
            ruler_sha256="d" * 64,
            task_map={"0": "src-a:0"},
        ).save(outputs / MANIFEST_FILENAME)
        export_path = tmp_path / "export.json"
        export_path.write_text(
            json.dumps([_task(0, [_annotation(7, "5 - Preserva.")])]), encoding="utf-8"
        )
        summary = run_pull_annotation("run-a", export_file=export_path, base_dir=tmp_path)
        assert summary.project_id == 0

    def test_requires_a_client_or_a_file(self, pushed: Path) -> None:
        with pytest.raises(ValueError, match="export_file"):
            run_pull_annotation("run-a", base_dir=pushed)
