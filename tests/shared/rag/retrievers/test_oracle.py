"""Tests for shared/rag/retrievers/oracle.py (OracleRetriever gold-context arm)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from arandu.shared.rag.protocol import Retriever, SourceRetriever
from arandu.shared.rag.retrievers.oracle import OracleRetriever

if TYPE_CHECKING:
    from pathlib import Path


def _pair(question: str, context: str, chunk_id: str | None) -> dict:
    return {
        "question": question,
        "answer": "Resposta.",
        "context": context,
        "question_type": "factual",
        "confidence": 0.9,
        "bloom_level": "remember",
        "chunk_id": chunk_id,
    }


def _write_cep(cep_dir: Path, source_file_id: str, pairs: list[dict]) -> None:
    cep_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "source_file_id": source_file_id,
        "source_filename": f"{source_file_id}.mp4",
        "transcription_text": "Texto.",
        "chunker_id": "cep_4k",
        "qa_pairs": pairs,
        "model_id": "qwen3:14b",
        "provider": "ollama",
        "language": "pt",
        "total_pairs": len(pairs),
        "bloom_distribution": {},
    }
    (cep_dir / f"{source_file_id}_cep_qa.json").write_text(json.dumps(record))


class TestOracleRetriever:
    def test_retriever_id_is_oracle(self) -> None:
        assert OracleRetriever({}).retriever_id == "oracle"

    def test_satisfies_source_retriever_not_retriever(self) -> None:
        # Keyed on provenance, so it must not pass as a question-keyed arm.
        retriever = OracleRetriever({})
        assert isinstance(retriever, SourceRetriever)
        assert not isinstance(retriever, Retriever)

    def test_returns_generation_context_as_prose_payload(self) -> None:
        retriever = OracleRetriever({"f:chk_0:0": ("chk_0", "A enchente de 1941.")})
        passages = retriever.retrieve_source("f:chk_0:0", top_k=10)
        assert len(passages) == 1
        passage = passages[0]
        assert passage.chunk_id == "chk_0"
        assert passage.rank == 0
        assert passage.payload == "A enchente de 1941."
        assert passage.payload_is_prose is True
        assert passage.retriever_meta == {"source_qa_pair_id": "f:chk_0:0"}

    def test_top_k_zero_returns_empty(self) -> None:
        retriever = OracleRetriever({"f:chk_0:0": ("chk_0", "Contexto.")})
        assert retriever.retrieve_source("f:chk_0:0", top_k=0) == []

    def test_unknown_id_returns_empty_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            assert OracleRetriever({}).retrieve_source("missing:x:0", top_k=5) == []
        assert any("missing:x:0" in r.message for r in caplog.records)


class TestFromCepDir:
    def test_indexes_every_pair_by_composite_id(self, tmp_path: Path) -> None:
        cep_dir = tmp_path / "cep"
        _write_cep(
            cep_dir,
            "src_a",
            [_pair("P1?", "Contexto A.", "chk_a"), _pair("P1?", "Contexto B.", "chk_b")],
        )
        retriever = OracleRetriever.from_cep_dir(cep_dir)
        # Same question text, distinct sources: the id disambiguates.
        assert retriever.retrieve_source("src_a:chk_a:0", 5)[0].payload == "Contexto A."
        assert retriever.retrieve_source("src_a:chk_b:1", 5)[0].payload == "Contexto B."

    def test_pair_without_chunk_id_falls_back_to_qa_pair_id(self, tmp_path: Path) -> None:
        cep_dir = tmp_path / "cep"
        _write_cep(cep_dir, "src_a", [_pair("P?", "Contexto.", None)])
        passage = OracleRetriever.from_cep_dir(cep_dir).retrieve_source("src_a:none:0", 5)[0]
        assert passage.chunk_id == "src_a:none:0"

    def test_unreadable_file_is_skipped(self, tmp_path: Path) -> None:
        cep_dir = tmp_path / "cep"
        _write_cep(cep_dir, "src_a", [_pair("P?", "Contexto.", "chk_a")])
        (cep_dir / "broken.json").write_text("{not json")
        retriever = OracleRetriever.from_cep_dir(cep_dir)
        assert retriever.retrieve_source("src_a:chk_a:0", 5)

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="CEP outputs not found"):
            OracleRetriever.from_cep_dir(tmp_path / "absent")
