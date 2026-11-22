"""Tests for shared/chunking/schemas.py — Chunk and ChunkSet."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path  # noqa: TC003 — used as a parameter type at runtime

import pytest
from pydantic import ValidationError

from arandu.shared.chunking.schemas import Chunk, ChunkSet


def _chunk(
    chunk_id: str = "abc1234567890def",
    source_file_id: str = "src1",
    chunker_id: str = "cep_4k",
    start_char: int = 0,
    end_char: int = 100,
    token_count: int | None = None,
    tokenizer_id: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_file_id=source_file_id,
        chunker_id=chunker_id,
        start_char=start_char,
        end_char=end_char,
        token_count=token_count,
        tokenizer_id=tokenizer_id,
    )


class TestChunk:
    def test_minimal_chunk_validates(self) -> None:
        chunk = _chunk()
        assert chunk.start_char == 0
        assert chunk.end_char == 100
        assert chunk.token_count is None
        assert chunk.tokenizer_id is None

    def test_end_char_must_exceed_start_char(self) -> None:
        with pytest.raises(ValidationError, match="end_char must be > start_char"):
            _chunk(start_char=100, end_char=100)

    def test_end_char_less_than_start_char_rejected(self) -> None:
        with pytest.raises(ValidationError, match="end_char must be > start_char"):
            _chunk(start_char=100, end_char=50)

    def test_negative_start_char_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(start_char=-1, end_char=50)

    def test_chunk_carries_no_text_field(self) -> None:
        chunk = _chunk()
        assert "text" not in chunk.model_dump()


class TestChunkSet:
    def _build(self, views: dict[str, list[Chunk]] | None = None) -> ChunkSet:
        return ChunkSet(
            source_file_id="src1",
            source_filename="entrevista.mp4",
            source_text_sha256="a" * 64,
            views=views
            or {
                "cep_4k": [_chunk(chunk_id="cep00000000000001", start_char=0, end_char=4000)],
                "bm25_512t": [
                    _chunk(
                        chunk_id="bm2500000000000001",
                        chunker_id="bm25_512t",
                        start_char=0,
                        end_char=1820,
                        token_count=512,
                        tokenizer_id="qwen2",
                    ),
                    _chunk(
                        chunk_id="bm2500000000000002",
                        chunker_id="bm25_512t",
                        start_char=1640,
                        end_char=3460,
                        token_count=512,
                        tokenizer_id="qwen2",
                    ),
                ],
            },
            generated_at=datetime(2026, 5, 13, 12, 0, 0),
        )

    def test_view_returns_chunks_for_known_id(self) -> None:
        cs = self._build()
        assert len(cs.view("cep_4k")) == 1
        assert len(cs.view("bm25_512t")) == 2

    def test_view_raises_keyerror_with_available_views_listed(self) -> None:
        cs = self._build()
        with pytest.raises(KeyError, match="bm25_4k"):
            cs.view("bm25_4k")

    def test_lookup_finds_chunk_across_views(self) -> None:
        cs = self._build()
        chunk = cs.lookup("bm2500000000000002")
        assert chunk.start_char == 1640

    def test_lookup_raises_for_unknown_chunk_id(self) -> None:
        cs = self._build()
        with pytest.raises(KeyError, match="missing"):
            cs.lookup("missing")

    def test_overlapping_returns_chunks_overlapping_range(self) -> None:
        cs = self._build()
        # Range [1700, 1900) overlaps the second bm25 chunk (1640..3460), not the first (0..1820)?
        # Actually it overlaps BOTH because chunk1 [0..1820) includes 1700..1820,
        # and chunk2 [1640..3460) includes 1700..1900.
        hits = cs.overlapping(1700, 1900, "bm25_512t")
        assert len(hits) == 2

    def test_overlapping_returns_empty_when_no_overlap(self) -> None:
        cs = self._build()
        assert cs.overlapping(10000, 20000, "bm25_512t") == []

    def test_overlapping_unknown_view_raises(self) -> None:
        cs = self._build()
        with pytest.raises(KeyError):
            cs.overlapping(0, 100, "nonexistent")

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        cs = self._build()
        path = tmp_path / "chunks.json"
        cs.save(path)
        loaded = ChunkSet.load(path)
        assert loaded.source_file_id == "src1"
        assert loaded.source_text_sha256 == "a" * 64
        assert set(loaded.views) == {"cep_4k", "bm25_512t"}
        assert loaded.lookup("cep00000000000001").end_char == 4000

    def test_serialization_omits_text_fields(self) -> None:
        cs = self._build()
        dumped = cs.model_dump_json()
        # Sanity: no spurious 'text' keys leak through
        assert '"text"' not in dumped
