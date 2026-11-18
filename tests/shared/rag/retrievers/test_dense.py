"""Tests for shared/rag/retrievers/dense.py: windowed dense retrieval over chunks.

A bag-of-words fake encoder stands in for the sentence encoder, so the tests
never download a model.
"""

from __future__ import annotations

import json
import re
import zlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arandu.shared.chunking.resolver import ChunkResolver
from arandu.shared.chunking.schemas import Chunk
from arandu.shared.rag.protocol import Retriever
from arandu.shared.rag.retrievers.dense import (
    EMBEDDINGS_FILENAME,
    MANIFEST_FILENAME,
    SUBPASSAGES_FILENAME,
    DenseRetriever,
    resolve_window,
    split_windows,
)
from arandu.shared.rag.schemas import RetrievalQuery

if TYPE_CHECKING:
    from pathlib import Path

CHUNKER_ID = "cep_4k"
DIM = 256


class FakeEncoder:
    """Whitespace tokens; each text embeds as its normalized bag of hashed words."""

    model_name = "fake/bow"
    num_special_tokens = 2

    def __init__(
        self, max_seq_length: int = 6, query_prompt: str = "", passage_prompt: str = ""
    ) -> None:
        self.max_seq_length = max_seq_length
        self.query_prompt = query_prompt
        self.passage_prompt = passage_prompt
        self.encoded: list[list[str]] = []

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def encode(self, texts: list[str], *, query: bool) -> np.ndarray:
        prompt = self.query_prompt if query else self.passage_prompt
        texts = [prompt + t for t in texts]
        self.encoded.append(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, zlib.crc32(word.encode()) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


# Chunk 0 mentions the flood only in its second window; chunk 1 is unrelated;
# chunk 2 mentions the river.
TEXTS = [
    "alfa beta gama delta epsilon enchente itaqui",
    "maria contou sobre a casa",
    "rio uruguai subiu muito",
]


def _corpus(tmp_path: Path) -> tuple[list[Chunk], ChunkResolver, Path]:
    full_text = " ".join(TEXTS)
    chunks: list[Chunk] = []
    pos = 0
    for i, text in enumerate(TEXTS):
        chunks.append(
            Chunk(
                chunk_id=f"chk{i}",
                source_file_id="src",
                chunker_id=CHUNKER_ID,
                start_char=pos,
                end_char=pos + len(text),
            )
        )
        pos += len(text) + 1
    return chunks, ChunkResolver(text_loader=lambda _fid: full_text), tmp_path / "dense"


def _built(
    tmp_path: Path, encoder: FakeEncoder | None = None, **kwargs: object
) -> tuple[DenseRetriever, FakeEncoder, Path]:
    chunks, resolver, index_dir = _corpus(tmp_path)
    encoder = encoder or FakeEncoder()
    whole = bool(kwargs.get("whole_chunk", False))
    DenseRetriever.build_index(chunks, resolver, index_dir, CHUNKER_ID, encoder, whole_chunk=whole)
    retriever = DenseRetriever(index_dir, encoder, CHUNKER_ID, **kwargs)  # type: ignore[arg-type]
    return retriever, encoder, index_dir


class TestSplitWindows:
    def test_overlapping_windows_cover_every_token(self) -> None:
        spans = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        assert split_windows(spans, 2, 1) == [(0, 3), (2, 5), (4, 7), (6, 9)]

    def test_no_overlap(self) -> None:
        spans = [(0, 1), (2, 3), (4, 5)]
        assert split_windows(spans, 2, 0) == [(0, 3), (4, 5)]

    def test_short_text_is_one_window(self) -> None:
        assert split_windows([(0, 4), (5, 9)], 128, 32) == [(0, 9)]

    def test_empty(self) -> None:
        assert split_windows([], 4, 1) == []


class TestResolveWindow:
    def test_default_is_encoder_input_minus_special_tokens(self) -> None:
        assert resolve_window(FakeEncoder(max_seq_length=128), None, 0.25) == (126, 32)

    def test_explicit_window_counts_special_tokens(self) -> None:
        assert resolve_window(FakeEncoder(max_seq_length=512), 256, 0.0) == (254, 0)

    def test_passage_prompt_takes_window_room(self) -> None:
        enc = FakeEncoder(max_seq_length=128, passage_prompt="passage: ")
        assert resolve_window(enc, None, 0.0) == (125, 0)

    def test_window_beyond_encoder_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            resolve_window(FakeEncoder(max_seq_length=128), 129, 0.25)

    def test_overlap_ratio_bounds(self) -> None:
        with pytest.raises(ValueError, match="overlap_ratio"):
            resolve_window(FakeEncoder(), None, 1.0)


class TestBuildIndex:
    def test_writes_files_and_manifest(self, tmp_path: Path) -> None:
        _, _, index_dir = _built(tmp_path)
        for name in (EMBEDDINGS_FILENAME, SUBPASSAGES_FILENAME, MANIFEST_FILENAME):
            assert (index_dir / name).exists()
        manifest = json.loads((index_dir / MANIFEST_FILENAME).read_text())
        assert manifest["chunker_id"] == CHUNKER_ID
        assert manifest["encoder_model"] == "fake/bow"
        # max_seq_length 6 minus 2 special tokens; 25% of 4 rounds to 1.
        assert manifest["window_tokens"] == 6
        assert manifest["content_tokens"] == 4
        assert manifest["overlap_tokens"] == 1
        assert manifest["chunks_indexed"] == 3
        # 7 tokens -> 2 windows; 5 tokens -> 2 windows; 4 tokens -> 1 window.
        assert manifest["subpassages_indexed"] == 5
        assert manifest["chunk_ids"] == ["chk0", "chk1", "chk2"]
        assert len(manifest["sha256"]) == 64

    def test_embeddings_are_unit_rows(self, tmp_path: Path) -> None:
        _, _, index_dir = _built(tmp_path)
        emb = np.load(index_dir / EMBEDDINGS_FILENAME)
        assert emb.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-5)

    def test_subpassages_are_slices_of_the_chunk(self, tmp_path: Path) -> None:
        _, _, index_dir = _built(tmp_path)
        subs = json.loads((index_dir / SUBPASSAGES_FILENAME).read_text())
        first = [s for s in subs if s["chunk_id"] == "chk0"]
        assert [s["text"] for s in first] == [
            "alfa beta gama delta",
            "delta epsilon enchente itaqui",
        ]
        for s in first:
            assert TEXTS[0][s["start_char"] : s["end_char"]] == s["text"]

    def test_empty_chunks_rejected(self, tmp_path: Path) -> None:
        _, resolver, index_dir = _corpus(tmp_path)
        with pytest.raises(ValueError, match="empty chunks"):
            DenseRetriever.build_index([], resolver, index_dir, CHUNKER_ID, FakeEncoder())


class TestLoadValidation:
    def test_satisfies_protocol(self, tmp_path: Path) -> None:
        retriever, _, _ = _built(tmp_path)
        assert isinstance(retriever, Retriever)
        assert retriever.retriever_id == "dense_cep_4k"

    def test_missing_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="missing"):
            DenseRetriever(tmp_path / "nope", FakeEncoder(), CHUNKER_ID)

    def test_window_mismatch_raises(self, tmp_path: Path) -> None:
        _, _, index_dir = _built(tmp_path)
        with pytest.raises(ValueError, match="window_tokens mismatch"):
            DenseRetriever(index_dir, FakeEncoder(), CHUNKER_ID, window_tokens=5)

    def test_passage_prompt_mismatch_raises(self, tmp_path: Path) -> None:
        _, _, index_dir = _built(tmp_path)
        with pytest.raises(ValueError, match="passage_prompt mismatch"):
            DenseRetriever(index_dir, FakeEncoder(passage_prompt="passage: "), CHUNKER_ID)

    def test_encoder_mismatch_raises(self, tmp_path: Path) -> None:
        _, _, index_dir = _built(tmp_path)
        other = FakeEncoder()
        other.model_name = "fake/other"
        with pytest.raises(ValueError, match="encoder_model mismatch"):
            DenseRetriever(index_dir, other, CHUNKER_ID)

    def test_chunker_mismatch_raises(self, tmp_path: Path) -> None:
        _, _, index_dir = _built(tmp_path)
        with pytest.raises(ValueError, match="chunker_id mismatch"):
            DenseRetriever(index_dir, FakeEncoder(), "bm25_512t")

    def test_tampered_index_raises(self, tmp_path: Path) -> None:
        _, _, index_dir = _built(tmp_path)
        np.save(index_dir / EMBEDDINGS_FILENAME, np.zeros((5, DIM), dtype=np.float32))
        with pytest.raises(ValueError, match="sha256 mismatch"):
            DenseRetriever(index_dir, FakeEncoder(), CHUNKER_ID)

    def test_unknown_unit_rejected(self, tmp_path: Path) -> None:
        _, _, index_dir = _built(tmp_path)
        with pytest.raises(ValueError, match="unit"):
            DenseRetriever(index_dir, FakeEncoder(), CHUNKER_ID, unit="sentence")  # type: ignore[arg-type]


class TestRetrieveParent:
    def test_ranks_parent_by_best_subpassage(self, tmp_path: Path) -> None:
        retriever, _, _ = _built(tmp_path)
        passages = retriever.retrieve(RetrievalQuery(question="enchente itaqui"), top_k=3)
        top = passages[0]
        assert top.chunk_id == "chk0"
        assert top.rank == 0
        assert top.payload is None  # resolved by chunk_id, like BM25
        # The flood is in chunk 0's second window.
        assert top.retriever_meta["best_subpassage"] == 1
        assert top.retriever_meta["best_subpassage_score"] == pytest.approx(top.score)

    def test_each_chunk_returned_once(self, tmp_path: Path) -> None:
        retriever, _, _ = _built(tmp_path)
        passages = retriever.retrieve(RetrievalQuery(question="delta"), top_k=10)
        ids = [p.chunk_id for p in passages]
        assert len(ids) == len(set(ids)) == 3
        assert [p.rank for p in passages] == [0, 1, 2]
        scores = [p.score for p in passages]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_caps_and_zero(self, tmp_path: Path) -> None:
        retriever, _, _ = _built(tmp_path)
        assert len(retriever.retrieve(RetrievalQuery(question="rio"), top_k=1)) == 1
        assert retriever.retrieve(RetrievalQuery(question="rio"), top_k=0) == []

    def test_reads_only_question(self, tmp_path: Path) -> None:
        retriever, encoder, _ = _built(tmp_path)
        retriever.retrieve(RetrievalQuery(question="rio", source_qa_pair_id="x:chk0:0"), top_k=1)
        assert encoder.encoded[-1] == ["rio"]

    def test_record_meta_names_encoder(self, tmp_path: Path) -> None:
        retriever, _, _ = _built(tmp_path)
        assert retriever.last_retrieve_meta["encoder_model"] == "fake/bow"
        assert retriever.last_retrieve_meta["unit"] == "parent"


class TestPrompts:
    def test_query_prompt_only_on_questions(self, tmp_path: Path) -> None:
        encoder = FakeEncoder(query_prompt="Instruct: find\nQuery: ")
        retriever, _, _ = _built(tmp_path, encoder)
        assert not any(t.startswith("Instruct") for t in encoder.encoded[0])
        retriever.retrieve(RetrievalQuery(question="rio"), top_k=1)
        assert encoder.encoded[-1] == ["Instruct: find\nQuery: rio"]
        assert retriever.last_retrieve_meta["query_prompt"] == "Instruct: find\nQuery: "


class TestWholeChunk:
    def test_one_subpassage_per_chunk(self, tmp_path: Path) -> None:
        retriever, _, index_dir = _built(tmp_path, whole_chunk=True)
        manifest = json.loads((index_dir / MANIFEST_FILENAME).read_text())
        assert manifest["window_tokens"] == "whole"
        assert manifest["subpassages_indexed"] == 3
        # With 2 special tokens, chunks of 7 and 5 tokens exceed the 6-token input.
        assert manifest["chunks_truncated"] == 2
        subs = json.loads((index_dir / SUBPASSAGES_FILENAME).read_text())
        assert [s["text"] for s in subs] == TEXTS
        top = retriever.retrieve(RetrievalQuery(question="enchente"), top_k=1)[0]
        assert top.chunk_id == "chk0"

    def test_windowed_index_does_not_load_as_whole(self, tmp_path: Path) -> None:
        _, _, index_dir = _built(tmp_path)
        with pytest.raises(ValueError, match="window_tokens mismatch"):
            DenseRetriever(index_dir, FakeEncoder(), CHUNKER_ID, whole_chunk=True)


class TestRetrieveByEmbedding:
    def test_matches_retrieve(self, tmp_path: Path) -> None:
        retriever, encoder, _ = _built(tmp_path)
        q = encoder.encode(["enchente rio"], query=True)[0]
        assert retriever.retrieve_by_embedding(q, 3) == retriever.retrieve(
            RetrievalQuery(question="enchente rio"), top_k=3
        )


class TestRetrieveSubpassage:
    def test_returns_windows_inline(self, tmp_path: Path) -> None:
        retriever, _, _ = _built(tmp_path, unit="subpassage")
        passages = retriever.retrieve(RetrievalQuery(question="enchente itaqui"), top_k=2)
        top = passages[0]
        assert top.chunk_id == "chk0:sp1"
        assert top.payload == "delta epsilon enchente itaqui"
        assert top.payload_is_prose is True
        assert top.retriever_meta["parent_chunk_id"] == "chk0"
        assert [p.rank for p in passages] == [0, 1]
