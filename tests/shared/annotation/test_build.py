"""The offline build: determinism, blinding, and the sign-off gate."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from arandu.shared.annotation.build import (
    CONFIG_FILENAME,
    MANIFEST_FILENAME,
    TASKS_FILENAME,
    run_build_annotation,
    shuffle_order,
)
from arandu.shared.annotation.ruler import RULER_PATH, RulerNotSignedOffError
from arandu.shared.annotation.schemas import AnnotationManifest
from arandu.shared.human_eval.schemas import SampleItem, SampleManifest

if TYPE_CHECKING:
    from pathlib import Path

FORBIDDEN_KEYS = {
    "pair_id",
    "source_file_id",
    "pair_index",
    "bloom_level",
    "emic_score",
    "cell_id",
    "slot_id",
}


def _item(index: int) -> SampleItem:
    return SampleItem(
        pair_id=f"src-{index // 2}:{index}",
        source_file_id=f"src-{index // 2}",
        pair_index=index,
        segment=f"segmento {index}",
        question=f"pergunta {index}",
        answer=f"resposta {index}",
        bloom_level=("remember", "understand", "analyze", "evaluate")[index % 4],
        emic_score=(index % 5) + 1,
        cell_id=f"{('remember', 'understand', 'analyze', 'evaluate')[index % 4]}:limpa",
        slot_id=index,
    )


@pytest.fixture
def sample_run(tmp_path: Path) -> Path:
    """A results tree with a populated human_eval stage (16 pairs)."""
    outputs = tmp_path / "run-a" / "human_eval" / "outputs"
    outputs.mkdir(parents=True)
    items = [_item(i) for i in range(16)]
    with (outputs / "sample.jsonl").open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json())
            fh.write("\n")
    SampleManifest(
        pipeline_id="run-a",
        seed=1,
        total_items=len(items),
        per_cell=2,
        cell_counts={},
        population_by_cell={},
        pool_sha256="c" * 64,
    ).save(outputs / "sample_manifest.json")
    return tmp_path


def _unsigned_ruler(tmp_path: Path) -> Path:
    data = yaml.safe_load(RULER_PATH.read_text(encoding="utf-8"))
    data["signed_off"] = False
    path = tmp_path / "unsigned.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


class TestShuffleOrder:
    def test_is_a_permutation(self) -> None:
        ids = [f"s:{i}" for i in range(20)]
        assert sorted(shuffle_order(ids, seed=3)) == sorted(ids)

    def test_same_seed_same_order(self) -> None:
        ids = [f"s:{i}" for i in range(20)]
        assert shuffle_order(ids, seed=3) == shuffle_order(ids, seed=3)

    def test_different_seed_different_order(self) -> None:
        ids = [f"s:{i}" for i in range(20)]
        assert shuffle_order(ids, seed=3) != shuffle_order(ids, seed=4)

    def test_order_is_independent_of_input_order(self) -> None:
        ids = [f"s:{i}" for i in range(20)]
        assert shuffle_order(ids, seed=3) == shuffle_order(list(reversed(ids)), seed=3)

    def test_it_actually_breaks_the_cell_grouping(self) -> None:
        """The sample arrives grouped by cell; the point is to destroy that."""
        ids = [f"s:{i}" for i in range(20)]
        assert shuffle_order(ids, seed=3) != ids


class TestSignOffGate:
    def test_unsigned_ruler_refuses_to_build(self, sample_run: Path, tmp_path: Path) -> None:
        with pytest.raises(RulerNotSignedOffError, match="anthropologist-validation"):
            run_build_annotation(
                "run-a", seed=5, base_dir=sample_run, ruler_path=_unsigned_ruler(tmp_path)
            )

    def test_unsigned_ruler_writes_no_tasks(self, sample_run: Path, tmp_path: Path) -> None:
        with pytest.raises(RulerNotSignedOffError):
            run_build_annotation(
                "run-a", seed=5, base_dir=sample_run, ruler_path=_unsigned_ruler(tmp_path)
            )
        assert not (sample_run / "run-a" / "annotation" / "outputs" / TASKS_FILENAME).exists()


class TestArtifacts:
    def test_writes_the_three_artifacts(self, sample_run: Path) -> None:
        run_build_annotation("run-a", seed=5, base_dir=sample_run)
        outputs = sample_run / "run-a" / "annotation" / "outputs"
        assert (outputs / CONFIG_FILENAME).exists()
        assert (outputs / TASKS_FILENAME).exists()
        assert (outputs / MANIFEST_FILENAME).exists()

    def test_task_count_matches_the_sample(self, sample_run: Path) -> None:
        manifest = run_build_annotation("run-a", seed=5, base_dir=sample_run)
        outputs = sample_run / "run-a" / "annotation" / "outputs"
        tasks = json.loads((outputs / TASKS_FILENAME).read_text(encoding="utf-8"))
        assert len(tasks) == 16
        assert manifest.total_items == 16

    def test_manifest_copies_provenance_from_the_sample(self, sample_run: Path) -> None:
        manifest = run_build_annotation("run-a", seed=5, base_dir=sample_run)
        assert manifest.pool_sha256 == "c" * 64
        assert manifest.per_cell == 2
        assert len(manifest.ruler_sha256) == 64
        assert manifest.project_id is None


class TestBlinding:
    def test_tasks_carry_only_the_four_allowed_keys(self, sample_run: Path) -> None:
        run_build_annotation("run-a", seed=5, base_dir=sample_run)
        outputs = sample_run / "run-a" / "annotation" / "outputs"
        tasks = json.loads((outputs / TASKS_FILENAME).read_text(encoding="utf-8"))
        for task in tasks:
            assert set(task.keys()) == {"data"}
            assert set(task["data"].keys()) == {"task_id", "segment", "question", "answer"}

    def test_no_forbidden_value_appears_anywhere_in_tasks_json(self, sample_run: Path) -> None:
        """Checked against the raw text, so a nested leak cannot hide."""
        run_build_annotation("run-a", seed=5, base_dir=sample_run)
        outputs = sample_run / "run-a" / "annotation" / "outputs"
        raw = (outputs / TASKS_FILENAME).read_text(encoding="utf-8")
        for key in FORBIDDEN_KEYS:
            assert key not in raw
        assert "src-0:0" not in raw


class TestJoin:
    def test_task_map_covers_every_task_exactly_once(self, sample_run: Path) -> None:
        manifest = run_build_annotation("run-a", seed=5, base_dir=sample_run)
        assert sorted(int(k) for k in manifest.task_map) == list(range(16))
        assert len(set(manifest.task_map.values())) == 16

    def test_task_map_matches_the_shuffled_payload(self, sample_run: Path) -> None:
        manifest = run_build_annotation("run-a", seed=5, base_dir=sample_run)
        outputs = sample_run / "run-a" / "annotation" / "outputs"
        tasks = json.loads((outputs / TASKS_FILENAME).read_text(encoding="utf-8"))
        for task in tasks:
            task_id = task["data"]["task_id"]
            pair_index = int(manifest.pair_id_for(task_id).split(":")[1])
            assert task["data"]["question"] == f"pergunta {pair_index}"


class TestDeterminism:
    def test_same_seed_reproduces_tasks_json_byte_for_byte(self, sample_run: Path) -> None:
        run_build_annotation("run-a", seed=5, base_dir=sample_run)
        outputs = sample_run / "run-a" / "annotation" / "outputs"
        first = (outputs / TASKS_FILENAME).read_bytes()
        run_build_annotation("run-a", seed=5, base_dir=sample_run)
        assert (outputs / TASKS_FILENAME).read_bytes() == first

    def test_different_seed_changes_the_order(self, sample_run: Path) -> None:
        run_build_annotation("run-a", seed=5, base_dir=sample_run)
        outputs = sample_run / "run-a" / "annotation" / "outputs"
        first = (outputs / TASKS_FILENAME).read_bytes()
        run_build_annotation("run-a", seed=6, base_dir=sample_run)
        assert (outputs / TASKS_FILENAME).read_bytes() != first


class TestErrors:
    def test_missing_sample_names_the_upstream_command(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="build-human-eval-sample"):
            run_build_annotation("absent", seed=5, base_dir=tmp_path)

    def test_count_mismatch_with_the_sample_manifest_fails(self, sample_run: Path) -> None:
        manifest_path = sample_run / "run-a" / "human_eval" / "outputs" / "sample_manifest.json"
        stale = SampleManifest.load(manifest_path)
        stale.total_items = 120
        stale.save(manifest_path)
        with pytest.raises(ValueError, match="120"):
            run_build_annotation("run-a", seed=5, base_dir=sample_run)

    def test_rebuild_after_push_refuses(self, sample_run: Path) -> None:
        """Rebuilding under a live project would desync the join."""
        run_build_annotation("run-a", seed=5, base_dir=sample_run)
        path = sample_run / "run-a" / "annotation" / "outputs" / MANIFEST_FILENAME
        pushed = AnnotationManifest.load(path)
        pushed.project_id = 42
        pushed.save(path)
        with pytest.raises(ValueError, match="42"):
            run_build_annotation("run-a", seed=5, base_dir=sample_run)
