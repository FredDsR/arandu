"""Tests for ``shared/rag/retrieve/settings.py`` — per-arm Pydantic settings."""

from __future__ import annotations

import pytest

from arandu.shared.rag.retrieve.settings import (
    ALL_ARMS,
    Bm25RetrieveSettings,
    KHopRetrieveSettings,
)


class TestArmCatalog:
    def test_known_arms(self) -> None:
        # The CLI's default arm set + the batch runner's validation both
        # key off this tuple. Drift would silently break either path.
        assert ALL_ARMS == ("bm25", "khop_passage", "khop_triple", "null")


class TestBm25Settings:
    def test_defaults(self) -> None:
        s = Bm25RetrieveSettings(_env_file=None)
        assert s.chunker_id == "cep_4k"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_BM25_CHUNKER_ID", "bm25_512t")
        assert Bm25RetrieveSettings().chunker_id == "bm25_512t"


class TestKHopSettings:
    def test_defaults(self) -> None:
        s = KHopRetrieveSettings(_env_file=None)
        assert s.k_hop == 2
        assert s.max_postings == 200
        assert s.keyword == "transcriptions.json"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_KHOP_K_HOP", "3")
        monkeypatch.setenv("ARANDU_KHOP_MAX_POSTINGS", "50")
        s = KHopRetrieveSettings()
        assert s.k_hop == 3
        assert s.max_postings == 50

    def test_invalid_k_hop_rejected(self) -> None:
        # k_hop=0 doesn't match any real retrieval pattern; reject at
        # settings-construction time so the error doesn't surface deep
        # in the retriever stack.
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            KHopRetrieveSettings(k_hop=0)
