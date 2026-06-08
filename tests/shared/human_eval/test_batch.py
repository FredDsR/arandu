"""Tests for the human-eval sample batch (spec §5)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from arandu.qa.schemas import QAPairCEP, QARecordCEP
from arandu.shared.emic.schemas import EmicScore, EmicSourceScores
from arandu.shared.human_eval.batch import run_build_sample_batch
from arandu.shared.human_eval.schemas import SampleItem, SampleManifest

if TYPE_CHECKING:
    from pathlib import Path

FRAME = ("remember", "understand", "analyze", "evaluate")


def _frame_specs(per_cell: int) -> list[tuple[str, int | None]]:
    """``per_cell`` duvidosa (score 2) + ``per_cell`` limpa (score 5) for each Bloom."""
    specs: list[tuple[str, int | None]] = []
    for bloom in FRAME:
        specs += [(bloom, 2)] * per_cell
        specs += [(bloom, 5)] * per_cell
    return specs


def _write_source(
    base: Path, pipeline_id: str, source_id: str, specs: list[tuple[str, int | None]]
) -> None:
    cep_outputs = base / pipeline_id / "cep" / "outputs"
    emic_outputs = base / pipeline_id / "emic_prepass" / "outputs"
    cep_outputs.mkdir(parents=True, exist_ok=True)
    emic_outputs.mkdir(parents=True, exist_ok=True)

    pairs = [
        QAPairCEP(
            question=f"q{i}",
            answer=f"a{i}",
            context=f"segment {i}",
            question_type="conceptual",
            confidence=0.9,
            bloom_level=bloom,
        )
        for i, (bloom, _score) in enumerate(specs)
    ]
    QARecordCEP(
        source_gdrive_id=source_id,
        source_filename=f"{source_id}.mp4",
        transcription_text="t",
        qa_pairs=pairs,
        model_id="m",
        provider="ollama",
        total_pairs=len(pairs),
    ).save(cep_outputs / f"{source_id}_cep_qa.json")

    scores = [
        EmicScore(pair_index=i, bloom_level=bloom, emic_score=score, rationale="r")
        for i, (bloom, score) in enumerate(specs)
    ]
    EmicSourceScores(
        source_file_id=source_id, source_filename=f"{source_id}.mp4", scores=scores
    ).save(emic_outputs / f"{source_id}_cep_qa.json")


def _load_sample(base: Path, pipeline_id: str) -> list[SampleItem]:
    path = base / pipeline_id / "human_eval" / "outputs" / "sample.jsonl"
    return [SampleItem.model_validate_json(line) for line in path.read_text().splitlines()]


class TestRunBuildSampleBatch:
    def test_happy_path_counts_and_manifest(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "run1", "s1", _frame_specs(2))

        manifest = run_build_sample_batch("run1", seed=42, base_dir=tmp_path, per_cell=2)

        assert manifest.total_items == 16  # 8 cells x 2
        assert len(manifest.cell_counts) == 8
        assert all(c == 2 for c in manifest.cell_counts.values())
        assert manifest.seed == 42
        assert manifest.pool_sha256  # provenance recorded

        sample = _load_sample(tmp_path, "run1")
        assert len(sample) == 16
        # run finalized as COMPLETED
        meta = json.loads(
            (tmp_path / "run1" / "human_eval" / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert meta["status"] == "completed"

    def test_excludes_out_of_frame_bloom_and_null_score(self, tmp_path: Path) -> None:
        specs = [*_frame_specs(2), ("apply", 5), ("create", 2), ("remember", None)]
        _write_source(tmp_path, "run2", "s1", specs)

        manifest = run_build_sample_batch("run2", seed=1, base_dir=tmp_path, per_cell=2)

        assert manifest.total_items == 16  # exclusions don't change the sample size
        assert manifest.excluded_none_score == 1
        assert manifest.excluded_bloom == {"apply": 1, "create": 1}

    def test_payload_is_blinded(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "run3", "s1", _frame_specs(2))
        run_build_sample_batch("run3", seed=1, base_dir=tmp_path, per_cell=2)
        item = _load_sample(tmp_path, "run3")[0]
        assert item.segment and item.question and item.answer  # payload present
        assert item.pair_id == f"{item.source_file_id}:{item.pair_index}"
        # SampleItem deliberately carries no tacit_inference / canonical scores
        dumped = item.model_dump()
        assert "tacit_inference" not in dumped
        assert "weighted_score" not in dumped

    def test_insufficient_cell_raises_and_marks_failed(self, tmp_path: Path) -> None:
        # remember:limpa gets only 1 limpa pair while per_cell=2.
        specs = _frame_specs(2)
        specs.remove(("remember", 5))  # drop one of the two remember-limpa pairs
        _write_source(tmp_path, "run4", "s1", specs)

        with pytest.raises(ValueError, match="remember:limpa"):
            run_build_sample_batch("run4", seed=1, base_dir=tmp_path, per_cell=2)

        meta = json.loads(
            (tmp_path / "run4" / "human_eval" / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert meta["status"] == "failed"

    def test_reproducible_across_runs(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "run5", "s1", _frame_specs(5))  # slack so selection is non-trivial
        run_build_sample_batch("run5", seed=7, base_dir=tmp_path, per_cell=2)
        first = [(i.pair_id, i.cell_id, i.slot_id) for i in _load_sample(tmp_path, "run5")]
        run_build_sample_batch("run5", seed=7, base_dir=tmp_path, per_cell=2)
        second = [(i.pair_id, i.cell_id, i.slot_id) for i in _load_sample(tmp_path, "run5")]
        assert first == second

    def test_missing_emic_stage_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Emic pre-pass outputs not found"):
            run_build_sample_batch("absent", seed=1, base_dir=tmp_path, per_cell=2)

    def test_missing_cep_stage_raises(self, tmp_path: Path) -> None:
        # emic outputs present, cep dir absent.
        emic_outputs = tmp_path / "run6" / "emic_prepass" / "outputs"
        emic_outputs.mkdir(parents=True)
        EmicSourceScores(
            source_file_id="s1",
            source_filename="s1.mp4",
            scores=[EmicScore(pair_index=0, bloom_level="remember", emic_score=2, rationale="r")],
        ).save(emic_outputs / "s1_cep_qa.json")

        with pytest.raises(FileNotFoundError, match="CEP outputs not found"):
            run_build_sample_batch("run6", seed=1, base_dir=tmp_path, per_cell=2)

    def test_empty_pool_raises_cause_specific(self, tmp_path: Path) -> None:
        # All approved pairs are out-of-frame Bloom or null-score -> empty frame.
        _write_source(tmp_path, "run8", "s1", [("apply", 5), ("create", 2), ("remember", None)])
        with pytest.raises(ValueError, match="No in-frame approved pairs"):
            run_build_sample_batch("run8", seed=1, base_dir=tmp_path, per_cell=2)

    def test_duplicate_pair_id_raises(self, tmp_path: Path) -> None:
        # Two emic files whose EmicSourceScores share the same source_file_id +
        # pair_index collide on pair_id (e.g. a stale duplicate emic output).
        emic_outputs = tmp_path / "run9" / "emic_prepass" / "outputs"
        cep_outputs = tmp_path / "run9" / "cep" / "outputs"
        emic_outputs.mkdir(parents=True)
        cep_outputs.mkdir(parents=True)
        for name in ("a", "b"):
            QARecordCEP(
                source_gdrive_id="dup",
                source_filename="dup.mp4",
                transcription_text="t",
                qa_pairs=[
                    QAPairCEP(
                        question="q",
                        answer="a",
                        context="seg",
                        question_type="conceptual",
                        confidence=0.9,
                        bloom_level="remember",
                    )
                ],
                model_id="m",
                provider="ollama",
                total_pairs=1,
            ).save(cep_outputs / f"{name}_cep_qa.json")
            EmicSourceScores(
                source_file_id="dup",
                source_filename="dup.mp4",
                scores=[
                    EmicScore(pair_index=0, bloom_level="remember", emic_score=2, rationale="r")
                ],
            ).save(emic_outputs / f"{name}_cep_qa.json")

        with pytest.raises(ValueError, match="Duplicate pair_id"):
            run_build_sample_batch("run9", seed=1, base_dir=tmp_path, per_cell=2)

    def test_pool_hash_changes_when_payload_drifts(self, tmp_path: Path) -> None:
        # Same pair ids/bloom/score but different CEP text -> different pool hash.
        _write_source(tmp_path, "runA", "s1", _frame_specs(2))
        h1 = run_build_sample_batch("runA", seed=1, base_dir=tmp_path, per_cell=2).pool_sha256
        # Rewrite the CEP records with mutated payload text, same indices/blooms.
        cep_outputs = tmp_path / "runA" / "cep" / "outputs"
        for cep_file in cep_outputs.glob("*_cep_qa.json"):
            rec = QARecordCEP.load(cep_file)
            for p in rec.qa_pairs:
                p.context = p.context + " (edited)"
            rec.save(cep_file)
        h2 = run_build_sample_batch("runA", seed=1, base_dir=tmp_path, per_cell=2).pool_sha256
        assert h1 != h2

    def test_manifest_roundtrips(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "run7", "s1", _frame_specs(2))
        run_build_sample_batch("run7", seed=3, base_dir=tmp_path, per_cell=2)
        manifest = SampleManifest.load(
            tmp_path / "run7" / "human_eval" / "outputs" / "sample_manifest.json"
        )
        assert manifest.seed == 3
        assert sum(manifest.cell_counts.values()) == 16
