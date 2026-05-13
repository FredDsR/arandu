"""Tests for shared/chunking/chonkie_adapter.py and protocol."""

from __future__ import annotations

import hashlib
from itertools import pairwise

from arandu.shared.chunking.chonkie_adapter import ChonkieAdapter, chunk_id_for
from arandu.shared.chunking.protocol import Chunker
from arandu.shared.chunking.schemas import Chunk


def _sample_text(n_chars: int = 8000) -> str:
    """Build a deterministic text long enough to exercise multi-chunk paths."""
    sentence = "Esta é uma frase de teste em português brasileiro com pontuação. "
    repeats = (n_chars // len(sentence)) + 2
    return (sentence * repeats)[:n_chars]


class TestChunkIdHelper:
    def test_chunk_id_is_stable(self) -> None:
        cid1 = chunk_id_for("src1", "cep_4k", 0, 4000)
        cid2 = chunk_id_for("src1", "cep_4k", 0, 4000)
        assert cid1 == cid2

    def test_chunk_id_changes_with_inputs(self) -> None:
        base = chunk_id_for("src1", "cep_4k", 0, 4000)
        assert chunk_id_for("src1", "cep_4k", 0, 4001) != base
        assert chunk_id_for("src1", "bm25_512t", 0, 4000) != base
        assert chunk_id_for("src2", "cep_4k", 0, 4000) != base

    def test_chunk_id_is_16_char_sha1_prefix(self) -> None:
        cid = chunk_id_for("src1", "cep_4k", 0, 4000)
        assert len(cid) == 16
        expected = hashlib.sha1(b"src1|cep_4k|0|4000").hexdigest()[:16]
        assert cid == expected


class TestChonkieAdapter:
    def test_adapter_satisfies_chunker_protocol(self) -> None:
        from chonkie import RecursiveChunker

        adapter = ChonkieAdapter(
            chunker_id="cep_4k", chonkie_chunker=RecursiveChunker(chunk_size=4000)
        )
        assert isinstance(adapter, Chunker)
        assert adapter.chunker_id == "cep_4k"

    def test_char_mode_chunker_emits_char_offsets(self) -> None:
        from chonkie import RecursiveChunker

        adapter = ChonkieAdapter(
            chunker_id="cep_4k", chonkie_chunker=RecursiveChunker(chunk_size=4000)
        )
        text = _sample_text(8000)
        chunks = adapter.chunk(text, source_file_id="src1")

        assert len(chunks) >= 2
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.source_file_id == "src1"
            assert c.chunker_id == "cep_4k"
            assert c.end_char > c.start_char
            assert 0 <= c.start_char < len(text)
            assert c.end_char <= len(text)

    def test_chunks_span_the_source_text(self) -> None:
        from chonkie import RecursiveChunker

        adapter = ChonkieAdapter(
            chunker_id="cep_4k", chonkie_chunker=RecursiveChunker(chunk_size=4000)
        )
        text = _sample_text(5000)
        chunks = adapter.chunk(text, source_file_id="src1")

        # Char ranges should be monotonic and start at 0
        assert chunks[0].start_char == 0
        for prev, nxt in pairwise(chunks):
            assert nxt.start_char >= prev.start_char

    def test_chunk_ids_are_stable_for_same_input(self) -> None:
        from chonkie import RecursiveChunker

        adapter1 = ChonkieAdapter(
            chunker_id="cep_4k", chonkie_chunker=RecursiveChunker(chunk_size=4000)
        )
        adapter2 = ChonkieAdapter(
            chunker_id="cep_4k", chonkie_chunker=RecursiveChunker(chunk_size=4000)
        )
        text = _sample_text(5000)
        chunks1 = adapter1.chunk(text, source_file_id="src1")
        chunks2 = adapter2.chunk(text, source_file_id="src1")
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_token_count_and_tokenizer_id_propagated_when_provided(self) -> None:
        from chonkie import RecursiveChunker

        adapter = ChonkieAdapter(
            chunker_id="bm25_4k",
            chonkie_chunker=RecursiveChunker(chunk_size=4000),
            tokenizer_id="character",
        )
        chunks = adapter.chunk(_sample_text(5000), source_file_id="src1")
        # chonkie's default tokenizer reports token_count; we propagate it
        for c in chunks:
            assert c.token_count is not None and c.token_count > 0
            assert c.tokenizer_id == "character"

    def test_empty_text_returns_empty_chunks(self) -> None:
        from chonkie import RecursiveChunker

        adapter = ChonkieAdapter(
            chunker_id="cep_4k", chonkie_chunker=RecursiveChunker(chunk_size=4000)
        )
        assert adapter.chunk("", source_file_id="src1") == []

    def test_offsets_resolve_to_text_round_trip(self) -> None:
        from chonkie import RecursiveChunker

        adapter = ChonkieAdapter(
            chunker_id="cep_4k", chonkie_chunker=RecursiveChunker(chunk_size=4000)
        )
        text = _sample_text(7000)
        chunks = adapter.chunk(text, source_file_id="src1")
        # For every chunk, the substring text[start:end] should be non-empty
        for c in chunks:
            assert text[c.start_char : c.end_char].strip() != ""
