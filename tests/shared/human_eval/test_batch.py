"""Tests for the human-eval sample batch (spec §5, revised 2026-08-19)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from arandu.qa.schemas import QAPairCEP, QARecordCEP
from arandu.shared.human_eval.batch import run_build_sample_batch
from arandu.shared.human_eval.schemas import SampleItem, SampleManifest
from arandu.shared.judge.schemas import JudgePipelineResult
from arandu.shared.schemas import SourceMetadata

if TYPE_CHECKING:
    from pathlib import Path

FRAME = ("remember", "understand", "analyze", "evaluate")


def _frame_specs(per_cell: int) -> list[str]:
    """``per_cell`` approved pairs for each Bloom level in the frame."""
    specs: list[str] = []
    for bloom in FRAME:
        specs += [bloom] * per_cell
    return specs


def _write_source(
    base: Path,
    pipeline_id: str,
    source_id: str,
    specs: list[str],
    *,
    approved: bool = True,
    metadata: SourceMetadata | None = None,
    metadata_enabled: bool = True,
) -> None:
    """Write one CEP record. No emic stage is written: the builder must not need it."""
    cep_outputs = base / pipeline_id / "cep" / "outputs"
    cep_outputs.mkdir(parents=True, exist_ok=True)
    pairs = [
        QAPairCEP(
            question=f"q{i}",
            answer=f"a{i}",
            context=f"segment {i}",
            question_type="conceptual",
            confidence=0.9,
            bloom_level=bloom,
            validation=JudgePipelineResult(stage_results={}, passed=approved),
        )
        for i, bloom in enumerate(specs)
    ]
    QARecordCEP(
        source_gdrive_id=source_id,
        source_filename=f"{source_id}.mp4",
        source_metadata=metadata,
        source_metadata_context_enabled=metadata_enabled,
        transcription_text="t",
        qa_pairs=pairs,
        model_id="m",
        provider="ollama",
        total_pairs=len(pairs),
    ).save(cep_outputs / f"{source_id}_cep_qa.json")


def _load_sample(base: Path, pipeline_id: str) -> list[SampleItem]:
    path = base / pipeline_id / "human_eval" / "outputs" / "sample.jsonl"
    return [SampleItem.model_validate_json(line) for line in path.read_text().splitlines()]


class TestRunBuildSampleBatch:
    def test_builds_with_no_emic_judge_stage_at_all(self, tmp_path: Path) -> None:
        """The regression that keeps the emic judge off the critical path.

        Nothing writes ``results/<id>/emic_judge/``, so a builder that still
        reads it fails here rather than weeks later, in the queue.
        """
        _write_source(tmp_path, "run1", "s1", _frame_specs(2))
        assert not (tmp_path / "run1" / "emic_judge").exists()

        manifest = run_build_sample_batch("run1", seed=42, base_dir=tmp_path, per_cell=2)

        assert manifest.total_items == 8  # 4 cells x 2
        assert set(manifest.cell_counts) == set(FRAME)
        assert all(c == 2 for c in manifest.cell_counts.values())
        assert manifest.seed == 42
        assert manifest.pool_sha256
        assert len(_load_sample(tmp_path, "run1")) == 8
        meta = json.loads(
            (tmp_path / "run1" / "human_eval" / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert meta["status"] == "completed"

    def test_provenance_points_at_the_cep_stage(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "run_prov", "s1", _frame_specs(2))
        run_build_sample_batch("run_prov", seed=1, base_dir=tmp_path, per_cell=2)
        meta = json.loads(
            (tmp_path / "run_prov" / "human_eval" / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert meta["input_source"].endswith("cep/outputs")

    def test_excludes_out_of_frame_bloom_and_counts_it(self, tmp_path: Path) -> None:
        specs = [*_frame_specs(2), "apply", "create"]
        _write_source(tmp_path, "run2", "s1", specs)

        manifest = run_build_sample_batch("run2", seed=1, base_dir=tmp_path, per_cell=2)

        assert manifest.total_items == 8  # exclusions don't change the sample size
        assert manifest.excluded_bloom == {"apply": 1, "create": 1}

    def test_frame_is_the_judge_approved_corpus(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "run3", "s1", _frame_specs(2))
        _write_source(tmp_path, "run3", "s2", _frame_specs(1), approved=False)

        manifest = run_build_sample_batch("run3", seed=1, base_dir=tmp_path, per_cell=2)

        assert manifest.excluded_not_approved == 4  # s2's four rejected pairs
        assert manifest.total_items == 8
        assert all(i.source_file_id == "s1" for i in _load_sample(tmp_path, "run3"))

    def test_population_by_cell_is_keyed_by_bloom_level(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "run_pop", "s1", _frame_specs(3))
        manifest = run_build_sample_batch("run_pop", seed=1, base_dir=tmp_path, per_cell=2)
        assert manifest.population_by_cell == dict.fromkeys(FRAME, 3)

    def test_payload_is_blinded(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "run4", "s1", _frame_specs(2))
        run_build_sample_batch("run4", seed=1, base_dir=tmp_path, per_cell=2)
        item = _load_sample(tmp_path, "run4")[0]
        dumped = item.model_dump()
        assert "emic_score" not in dumped
        assert "cell_id" not in dumped
        assert "tacit_inference" not in dumped
        assert dumped["segment"].startswith("segment ")

    def test_insufficient_cell_raises_and_marks_the_run_failed(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "run5", "s1", _frame_specs(1))
        with pytest.raises(ValueError, match="required"):
            run_build_sample_batch("run5", seed=1, base_dir=tmp_path, per_cell=2)
        meta = json.loads(
            (tmp_path / "run5" / "human_eval" / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert meta["status"] == "failed"

    def test_reproducible_across_runs(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "a", "s1", _frame_specs(5))
        _write_source(tmp_path, "b", "s1", _frame_specs(5))
        run_build_sample_batch("a", seed=77, base_dir=tmp_path, per_cell=2)
        run_build_sample_batch("b", seed=77, base_dir=tmp_path, per_cell=2)
        assert [i.pair_id for i in _load_sample(tmp_path, "a")] == [
            i.pair_id for i in _load_sample(tmp_path, "b")
        ]

    def test_missing_cep_stage_names_the_generate_command(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="generate-cep-qa"):
            run_build_sample_batch("absent", seed=1, base_dir=tmp_path, per_cell=2)

    def test_empty_pool_raises_cause_specific(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "run6", "s1", _frame_specs(2), approved=False)
        with pytest.raises(ValueError, match="not judge-approved"):
            run_build_sample_batch("run6", seed=1, base_dir=tmp_path, per_cell=2)

    def test_duplicate_pair_id_raises(self, tmp_path: Path) -> None:
        """Two CEP files sharing a source_file_id would collide the join key."""
        _write_source(tmp_path, "run7", "s1", _frame_specs(2))
        cep_outputs = tmp_path / "run7" / "cep" / "outputs"
        record = QARecordCEP.load(cep_outputs / "s1_cep_qa.json")
        record.save(cep_outputs / "s1_copy_cep_qa.json")
        with pytest.raises(ValueError, match="Duplicate pair_id"):
            run_build_sample_batch("run7", seed=1, base_dir=tmp_path, per_cell=2)

    def test_pool_hash_changes_when_payload_drifts(self, tmp_path: Path) -> None:
        """The digest covers the payload, so a CEP regeneration is detectable."""
        _write_source(tmp_path, "run8", "s1", _frame_specs(3))
        first = run_build_sample_batch("run8", seed=1, base_dir=tmp_path, per_cell=2)

        _write_source(tmp_path, "run9", "s1", _frame_specs(3))
        cep_path = tmp_path / "run9" / "cep" / "outputs" / "s1_cep_qa.json"
        record = QARecordCEP.load(cep_path)
        record.qa_pairs[0].answer = "regenerated answer"
        record.save(cep_path)
        second = run_build_sample_batch("run9", seed=1, base_dir=tmp_path, per_cell=2)

        assert first.pool_sha256 != second.pool_sha256

    def test_manifest_roundtrips(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "run10", "s1", _frame_specs(2))
        manifest = run_build_sample_batch("run10", seed=1, base_dir=tmp_path, per_cell=2)
        path = tmp_path / "run10" / "human_eval" / "outputs" / "sample_manifest.json"
        assert SampleManifest.load(path) == manifest


class TestSourceMetadataReachesTheSample:
    """Issue #173: the annotator sees the same metadata the generator had.

    Rendered here, at pool construction, because the CEP record is the last
    place the metadata and the gate are both in hand: the annotation build reads
    only ``sample.jsonl``.
    """

    def test_metadata_lines_are_carried_on_every_item(self, tmp_path: Path) -> None:
        _write_source(
            tmp_path,
            "run-md",
            "s1",
            _frame_specs(1),
            metadata=SourceMetadata(participant_name="Aida", location="DOQUINHAS"),
        )

        run_build_sample_batch("run-md", seed=1, base_dir=tmp_path, per_cell=1)

        for item in _load_sample(tmp_path, "run-md"):
            assert item.metadata == "- Participante: Aida\n- Local: DOQUINHAS"

    def test_no_header_line_is_carried(self, tmp_path: Path) -> None:
        """The canvas titles the block itself; a second heading would be noise."""
        _write_source(
            tmp_path, "run-hdr", "s1", _frame_specs(1), metadata=SourceMetadata(location="X")
        )

        run_build_sample_batch("run-hdr", seed=1, base_dir=tmp_path, per_cell=1)

        assert "Metadados da Entrevista:" not in _load_sample(tmp_path, "run-hdr")[0].metadata

    def test_empty_when_generation_had_metadata_disabled(self, tmp_path: Path) -> None:
        """Same gate as the judge: neither side sees what generation never injected."""
        _write_source(
            tmp_path,
            "run-off",
            "s1",
            _frame_specs(1),
            metadata=SourceMetadata(participant_name="Aida"),
            metadata_enabled=False,
        )

        run_build_sample_batch("run-off", seed=1, base_dir=tmp_path, per_cell=1)

        assert all(item.metadata == "" for item in _load_sample(tmp_path, "run-off"))

    def test_each_source_keeps_its_own_metadata(self, tmp_path: Path) -> None:
        _write_source(
            tmp_path,
            "run-two",
            "s1",
            _frame_specs(1),
            metadata=SourceMetadata(participant_name="Aida"),
        )
        _write_source(
            tmp_path,
            "run-two",
            "s2",
            _frame_specs(1),
            metadata=SourceMetadata(participant_name="Julia"),
        )

        run_build_sample_batch("run-two", seed=1, base_dir=tmp_path, per_cell=2)

        by_source = {
            item.source_file_id: item.metadata for item in _load_sample(tmp_path, "run-two")
        }
        assert by_source["s1"] == "- Participante: Aida"
        assert by_source["s2"] == "- Participante: Julia"
