"""Tests for shared/chunking/registry.py — get_chunker factory."""

from __future__ import annotations

import pytest

from arandu.shared.chunking.chonkie_adapter import ChonkieAdapter
from arandu.shared.chunking.protocol import Chunker
from arandu.shared.chunking.registry import KNOWN_CHUNKER_IDS, get_chunker


class TestGetChunker:
    @pytest.mark.parametrize("chunker_id", ["cep_4k", "bm25_4k", "nx_2k"])
    def test_char_mode_chunkers_resolve(self, chunker_id: str) -> None:
        chunker = get_chunker(chunker_id)
        assert isinstance(chunker, ChonkieAdapter)
        assert isinstance(chunker, Chunker)
        assert chunker.chunker_id == chunker_id

    def test_cep_4k_produces_chunks_under_4000_chars(self) -> None:
        chunker = get_chunker("cep_4k")
        text = "Esta é uma frase de teste. " * 500  # ~13500 chars
        chunks = chunker.chunk(text, source_file_id="src1")
        assert len(chunks) >= 2
        for c in chunks:
            assert c.end_char - c.start_char <= 4000

    def test_nx_2k_produces_chunks_under_2000_chars(self) -> None:
        chunker = get_chunker("nx_2k")
        text = "Esta é uma frase de teste. " * 500
        chunks = chunker.chunk(text, source_file_id="src1")
        for c in chunks:
            assert c.end_char - c.start_char <= 2000

    def test_unknown_chunker_id_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunker_id"):
            get_chunker("nonsense_view")

    def test_known_chunker_ids_constant_exposed(self) -> None:
        # Surface for CLI validation and tests
        assert "cep_4k" in KNOWN_CHUNKER_IDS
        assert "bm25_4k" in KNOWN_CHUNKER_IDS
        assert "nx_2k" in KNOWN_CHUNKER_IDS
