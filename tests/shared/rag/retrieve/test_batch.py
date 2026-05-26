"""Tests for ``shared/rag/retrieve/batch.py`` — the retrieve batch runner.

End-to-end with the null arm (always available, no on-disk artifacts).
Locks: checkpoint resume, output directory layout (cep/ vs nonanswerable/),
graceful per-arm failure (one arm down doesn't kill the batch), the
top_k validation surface.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from arandu.shared.rag.retrieve.batch import (
    CHECKPOINT_FILENAME,
    run_retrieve_batch,
)
from arandu.shared.rag.schemas import RetrievalRecord

if TYPE_CHECKING:
    from pathlib import Path


def _seed_cep(base: Path, pipeline_id: str = "run_x") -> None:
    """Populate a minimal CEP outputs tree with two QA pairs in one record."""
    cep_dir = base / pipeline_id / "cep" / "outputs"
    cep_dir.mkdir(parents=True)
    record = {
        "source_file_id": "src_a",
        "source_filename": "src_a.mp4",
        "transcription_text": "Texto do teste.",
        "chunker_id": "cep_4k",
        "qa_pairs": [
            {
                "question": "Pergunta 1?",
                "answer": "R1.",
                "context": "C1.",
                "question_type": "factual",
                "confidence": 0.9,
                "bloom_level": "remember",
                "chunk_id": "chk_00",
            },
            {
                "question": "Pergunta 2?",
                "answer": "R2.",
                "context": "C2.",
                "question_type": "factual",
                "confidence": 0.9,
                "bloom_level": "remember",
                "chunk_id": "chk_01",
            },
        ],
        "model_id": "qwen3:14b",
        "provider": "ollama",
        "language": "pt",
        "total_pairs": 2,
        "bloom_distribution": {},
    }
    (cep_dir / "src_a_cep_qa.json").write_text(json.dumps(record))


class TestRunRetrieveBatchNullArm:
    def test_null_arm_end_to_end(self, tmp_path: Path) -> None:
        _seed_cep(tmp_path)

        result = run_retrieve_batch(
            pipeline_id="run_x",
            arms=["null"],
            top_k=5,
            base_dir=tmp_path,
        )

        # Two CEP pairs xone arm = two records.
        assert result.retrievals_written == 2
        assert result.retrievals_failed == 0

        # Output layout: <outputs>/<arm>/<source>/<sanitized_qa_pair_id>.json
        outputs = tmp_path / "run_x" / "retrieve" / "outputs"
        cep_records = sorted((outputs / "null" / "cep").glob("*.json"))
        assert len(cep_records) == 2

        # Records round-trip via the RetrievalRecord schema and carry
        # the null arm's empty passage list.
        first = RetrievalRecord.load(cep_records[0])
        assert first.retriever_id == "null"
        assert first.passages == []
        assert first.is_answerable is True
        assert first.chunker_id == "cep_4k"

    def test_resume_skips_completed_tuples(self, tmp_path: Path) -> None:
        _seed_cep(tmp_path)

        first = run_retrieve_batch(
            pipeline_id="run_x",
            arms=["null"],
            top_k=5,
            base_dir=tmp_path,
        )
        assert first.retrievals_written == 2

        # Second invocation must replay nothing — checkpoint already
        # marks both (null, qa_pair) tuples as completed.
        second = run_retrieve_batch(
            pipeline_id="run_x",
            arms=["null"],
            top_k=5,
            base_dir=tmp_path,
        )
        assert second.retrievals_written == 0
        assert second.retrievals_resumed == 2

    def test_checkpoint_file_persisted(self, tmp_path: Path) -> None:
        # The checkpoint file is the resume contract; locking its
        # location prevents accidental relocation that would silently
        # disable resume.
        _seed_cep(tmp_path)
        run_retrieve_batch(
            pipeline_id="run_x",
            arms=["null"],
            top_k=5,
            base_dir=tmp_path,
        )
        ckpt_path = tmp_path / "run_x" / "retrieve" / CHECKPOINT_FILENAME
        assert ckpt_path.exists()


class TestRunRetrieveBatchValidation:
    def test_empty_arms_raises(self, tmp_path: Path) -> None:
        _seed_cep(tmp_path)
        with pytest.raises(ValueError, match="at least one arm"):
            run_retrieve_batch(pipeline_id="run_x", arms=[], top_k=5, base_dir=tmp_path)

    def test_top_k_zero_raises(self, tmp_path: Path) -> None:
        _seed_cep(tmp_path)
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            run_retrieve_batch(pipeline_id="run_x", arms=["null"], top_k=0, base_dir=tmp_path)

    def test_unknown_arm_raises(self, tmp_path: Path) -> None:
        _seed_cep(tmp_path)
        with pytest.raises(ValueError, match="Unknown arm"):
            run_retrieve_batch(
                pipeline_id="run_x",
                arms=["not_a_real_arm"],  # type: ignore[list-item]
                top_k=5,
                base_dir=tmp_path,
            )

    def test_missing_cep_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="CEP outputs not found"):
            run_retrieve_batch(
                pipeline_id="never_built",
                arms=["null"],
                top_k=5,
                base_dir=tmp_path,
            )


class TestAtlasRagMissingPrerequisites:
    def test_atlas_rag_arm_logs_missing_precompute_without_blocking_other_arms(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # atlas_rag is in ALL_ARMS so the batch-runner validation accepts
        # it. With no KG / no precompute on disk, the factory raises
        # FileNotFoundError pointing at `arandu kg-build-retriever-index`.
        # That error is caught by the per-arm build-with-logging path;
        # the run continues with other arms.
        import logging

        _seed_cep(tmp_path)
        with caplog.at_level(logging.ERROR, logger="arandu.shared.rag.retrieve.batch"):
            result = run_retrieve_batch(
                pipeline_id="run_x",
                arms=["atlas_rag", "null"],
                top_k=5,
                base_dir=tmp_path,
            )

        # null arm completes both questions; atlas_rag fails both.
        assert result.retrievals_written == 2
        assert result.retrievals_failed >= 2
        # The user gets a logged hint pointing at the build command.
        assert any("atlas-rag KG outputs not found" in r.message for r in caplog.records)


class TestPerArmFailureIsolation:
    def test_one_arm_missing_prereq_doesnt_block_others(self, tmp_path: Path) -> None:
        # BM25 arm without chunks fails at retriever construction; the
        # null arm should still run its questions and write records.
        # This is the "single bad arm doesn't kill the whole batch"
        # contract called out in the docstring.
        _seed_cep(tmp_path)

        result = run_retrieve_batch(
            pipeline_id="run_x",
            arms=["bm25", "null"],
            top_k=5,
            base_dir=tmp_path,
        )

        # BM25 fails for both questions (build failure counts each as failed),
        # but null completes both.
        assert result.retrievals_written == 2  # null x2 questions
        assert result.retrievals_failed >= 2  # bm25 x2 questions
        null_outputs = tmp_path / "run_x" / "retrieve" / "outputs" / "null" / "cep"
        assert len(list(null_outputs.glob("*.json"))) == 2
