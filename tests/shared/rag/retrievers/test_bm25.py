"""Tests for shared/rag/retrievers/bm25.py — BM25 sparse retrieval baseline (spec §4.3)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from arandu.shared.chunking.resolver import ChunkResolver
from arandu.shared.chunking.schemas import Chunk
from arandu.shared.rag.protocol import Retriever
from arandu.shared.rag.retrievers.bm25 import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    BM25Retriever,
)

if TYPE_CHECKING:
    from pathlib import Path


SOURCE_ID = "src_test_001"
CHUNKER_ID = "bm25_test"


def _build_corpus_fixture(tmp_path: Path) -> tuple[list[Chunk], ChunkResolver, Path]:
    """Build a 4-document corpus with known keyword overlap with sample queries."""
    sentences = [
        "A enchente de 2024 alagou completamente a cidade de Itaqui.",
        "Maria contou sobre a perda da casa e dos animais.",
        "O rio Uruguai subiu três metros em poucas horas.",
        "A enchente foi terrível e o rio Uruguai transbordou na madrugada.",
    ]
    full_text = " ".join(sentences)
    chunks: list[Chunk] = []
    pos = 0
    for i, sent in enumerate(sentences):
        chunks.append(
            Chunk(
                chunk_id=f"chk_{i:013d}",
                source_file_id=SOURCE_ID,
                chunker_id=CHUNKER_ID,
                start_char=pos,
                end_char=pos + len(sent),
            )
        )
        pos += len(sent) + 1  # +1 for the join space
    resolver = ChunkResolver(text_loader=lambda _fid: full_text)
    index_dir = tmp_path / "retrieval_indexes" / CHUNKER_ID / "bm25"
    return chunks, resolver, index_dir


class TestBM25BuildIndex:
    def test_creates_index_and_manifest_files(self, tmp_path: Path) -> None:
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        assert (index_dir / INDEX_FILENAME).exists()
        assert (index_dir / MANIFEST_FILENAME).exists()

    def test_manifest_contains_required_fields(self, tmp_path: Path) -> None:
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        manifest = json.loads((index_dir / MANIFEST_FILENAME).read_text())
        assert manifest["chunker_id"] == CHUNKER_ID
        assert manifest["language"] == "pt"
        assert manifest["chunks_indexed"] == 4
        assert "built_at" in manifest  # ISO timestamp
        assert manifest["chunk_ids"] == [c.chunk_id for c in chunks]

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        chunks, resolver, _ = _build_corpus_fixture(tmp_path)
        nested = tmp_path / "a" / "b" / "c"  # multiple missing levels
        BM25Retriever.build_index(chunks, resolver, nested, CHUNKER_ID, language="pt")
        assert (nested / INDEX_FILENAME).exists()

    def test_empty_chunks_raises(self, tmp_path: Path) -> None:
        _, resolver, index_dir = _build_corpus_fixture(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            BM25Retriever.build_index([], resolver, index_dir, CHUNKER_ID, language="pt")

    def test_invalid_language_raises(self, tmp_path: Path) -> None:
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        with pytest.raises(ValueError, match="Unsupported language"):
            BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="fr")


class TestBM25LoadAndConfig:
    def test_loads_index_via_constructor(self, tmp_path: Path) -> None:
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        retriever = BM25Retriever(index_dir=index_dir, chunker_id=CHUNKER_ID)
        assert retriever.retriever_id == f"bm25_{CHUNKER_ID}"

    def test_satisfies_retriever_protocol(self, tmp_path: Path) -> None:
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        retriever = BM25Retriever(index_dir=index_dir, chunker_id=CHUNKER_ID)
        assert isinstance(retriever, Retriever)

    def test_missing_index_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="BM25 index"):
            BM25Retriever(index_dir=tmp_path / "nonexistent", chunker_id=CHUNKER_ID)

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        # Only the pickle exists, no manifest
        idx_dir = tmp_path / "broken"
        idx_dir.mkdir()
        (idx_dir / INDEX_FILENAME).write_bytes(b"dummy")
        with pytest.raises(FileNotFoundError, match="BM25 index"):
            BM25Retriever(index_dir=idx_dir, chunker_id=CHUNKER_ID)

    def test_chunker_id_mismatch_raises(self, tmp_path: Path) -> None:
        # Index built for CHUNKER_ID; load with a different chunker_id must fail loudly.
        # Silent acceptance would corrupt retriever_id labels in RetrievalRecord.
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        with pytest.raises(ValueError, match="chunker_id mismatch"):
            BM25Retriever(index_dir=index_dir, chunker_id="bm25_other", language="pt")

    def test_language_mismatch_raises(self, tmp_path: Path) -> None:
        # Index tokenized as Portuguese; load with English would silently destroy ranking.
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        with pytest.raises(ValueError, match="language mismatch"):
            BM25Retriever(index_dir=index_dir, chunker_id=CHUNKER_ID, language="en")


class TestBM25Retrieve:
    def test_retrieve_returns_at_most_top_k(self, tmp_path: Path) -> None:
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        retriever = BM25Retriever(index_dir=index_dir, chunker_id=CHUNKER_ID)
        results = retriever.retrieve("enchente", top_k=2)
        assert len(results) == 2

    def test_retrieve_returns_ranked_zero_indexed(self, tmp_path: Path) -> None:
        # "Itaqui Maria" — each token appears in exactly one doc (idf > 0), so the
        # ranking is non-degenerate and scores monotonically decrease.
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        retriever = BM25Retriever(index_dir=index_dir, chunker_id=CHUNKER_ID)
        results = retriever.retrieve("Itaqui Maria", top_k=4)
        assert [p.rank for p in results] == [0, 1, 2, 3]
        scores = [p.score for p in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_retrieve_ranks_relevant_chunks_first(self, tmp_path: Path) -> None:
        # "Itaqui" appears only in chunk 0; "Maria" only in chunk 1.
        # A combined query must place those two above chunks 2 and 3 (which match neither).
        # (Using single-doc terms avoids BM25Okapi's idf=0 degeneracy at df=N/2.)
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        retriever = BM25Retriever(index_dir=index_dir, chunker_id=CHUNKER_ID)
        results = retriever.retrieve("Itaqui Maria", top_k=4)
        top_2_ids = {p.chunk_id for p in results[:2]}
        assert top_2_ids == {chunks[0].chunk_id, chunks[1].chunk_id}
        assert results[0].score > 0
        assert results[1].score > 0
        # Chunks 2 and 3 have neither term → score 0
        assert results[2].score == 0
        assert results[3].score == 0

    def test_retrieve_deterministic(self, tmp_path: Path) -> None:
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        retriever = BM25Retriever(index_dir=index_dir, chunker_id=CHUNKER_ID)
        results_a = retriever.retrieve("enchente rio", top_k=3)
        results_b = retriever.retrieve("enchente rio", top_k=3)
        assert [(p.chunk_id, p.rank, p.score) for p in results_a] == [
            (p.chunk_id, p.rank, p.score) for p in results_b
        ]

    def test_retrieve_top_k_greater_than_corpus_returns_all(self, tmp_path: Path) -> None:
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        retriever = BM25Retriever(index_dir=index_dir, chunker_id=CHUNKER_ID)
        results = retriever.retrieve("enchente", top_k=100)
        # top_k > corpus size; returns all 4 chunks (RetrievalRecord enforces <= top_k separately)
        assert len(results) == 4

    def test_retrieve_meta_records_score_method(self, tmp_path: Path) -> None:
        chunks, resolver, index_dir = _build_corpus_fixture(tmp_path)
        BM25Retriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, language="pt")
        retriever = BM25Retriever(index_dir=index_dir, chunker_id=CHUNKER_ID)
        results = retriever.retrieve("enchente", top_k=1)
        assert results[0].retriever_meta == {"score_method": "bm25_okapi"}
