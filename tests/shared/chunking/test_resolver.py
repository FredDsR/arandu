"""Tests for shared/chunking/resolver.py — ChunkResolver and SourceDriftError."""

from __future__ import annotations

import hashlib

import pytest

from arandu.shared.chunking.resolver import ChunkResolver, SourceDriftError
from arandu.shared.chunking.schemas import Chunk


def _make_chunk(start: int = 0, end: int = 100, source_file_id: str = "src1") -> Chunk:
    return Chunk(
        chunk_id=f"id{start:013d}",
        source_file_id=source_file_id,
        chunker_id="cep_4k",
        start_char=start,
        end_char=end,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestChunkResolver:
    def test_text_returns_substring(self) -> None:
        text = "Hello world, this is a transcription of some interview about floods."
        resolver = ChunkResolver(text_loader=lambda _: text)
        chunk = _make_chunk(start=6, end=11)
        assert resolver.text(chunk) == "world"

    def test_text_loader_cached_via_lru(self) -> None:
        text = "abcdefghij" * 100
        calls: list[str] = []

        def loader(file_id: str) -> str:
            calls.append(file_id)
            return text

        resolver = ChunkResolver(text_loader=loader)
        resolver.text(_make_chunk(0, 5))
        resolver.text(_make_chunk(5, 10))
        resolver.text(_make_chunk(2, 7))
        # Same source_file_id three times → loader called exactly once
        assert calls == ["src1"]

    def test_texts_resolves_multiple(self) -> None:
        text = "0123456789"
        resolver = ChunkResolver(text_loader=lambda _: text)
        chunks = [_make_chunk(0, 3), _make_chunk(3, 6), _make_chunk(6, 9)]
        assert resolver.texts(chunks) == ["012", "345", "678"]

    def test_drift_detected_when_expected_sha_mismatches(self) -> None:
        text = "the actual loaded text differs from what was chunked"
        wrong_sha = "f" * 64
        resolver = ChunkResolver(
            text_loader=lambda _: text,
            expected_sha256_by_file_id={"src1": wrong_sha},
        )
        with pytest.raises(SourceDriftError, match="src1"):
            resolver.text(_make_chunk(0, 5))

    def test_drift_passes_when_expected_sha_matches(self) -> None:
        text = "stable source text"
        resolver = ChunkResolver(
            text_loader=lambda _: text,
            expected_sha256_by_file_id={"src1": _sha256(text)},
        )
        assert resolver.text(_make_chunk(0, 6)) == "stable"

    def test_no_drift_check_when_lookup_missing_source(self) -> None:
        text = "any text"
        resolver = ChunkResolver(
            text_loader=lambda _: text,
            expected_sha256_by_file_id={"other_src": "f" * 64},
        )
        # source_file_id="src1" not in lookup → no drift check
        assert resolver.text(_make_chunk(0, 3)) == "any"

    def test_no_drift_check_when_lookup_is_none(self) -> None:
        text = "any text"
        resolver = ChunkResolver(text_loader=lambda _: text)
        assert resolver.text(_make_chunk(0, 3)) == "any"

    def test_drift_error_message_includes_expected_and_observed(self) -> None:
        text = "loaded text"
        wrong_sha = "f" * 64
        resolver = ChunkResolver(
            text_loader=lambda _: text,
            expected_sha256_by_file_id={"src1": wrong_sha},
        )
        with pytest.raises(SourceDriftError) as exc:
            resolver.text(_make_chunk(0, 5))
        msg = str(exc.value)
        assert wrong_sha[:8] in msg
        assert _sha256(text)[:8] in msg
