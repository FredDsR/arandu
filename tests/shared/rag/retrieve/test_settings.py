"""Tests for ``shared/rag/retrieve/settings.py`` — per-arm Pydantic settings."""

from __future__ import annotations

import pytest

from arandu.shared.rag.retrieve.settings import (
    ALL_ARMS,
    AtlasRagRetrieveSettings,
    Bm25RetrieveSettings,
    KHopRetrieveSettings,
)


class TestArmCatalog:
    def test_known_arms(self) -> None:
        # The CLI's default arm set + the batch runner's validation both
        # key off this tuple. Drift would silently break either path.
        # atlas_rag is recognized (so --arm atlas_rag routes to the
        # factory's clear "deferred PR" error) but excluded from the
        # CLI's default set until its LLM-client wiring lands.
        assert ALL_ARMS == ("bm25", "atlas_rag", "khop_passage", "khop_triple", "null")


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
        assert s.top_k_seeds == 50
        assert s.keyword == "transcriptions.json"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_KHOP_K_HOP", "3")
        monkeypatch.setenv("ARANDU_KHOP_TOP_K_SEEDS", "75")
        s = KHopRetrieveSettings()
        assert s.k_hop == 3
        assert s.top_k_seeds == 75

    def test_invalid_k_hop_rejected(self) -> None:
        # k_hop=0 doesn't match any real retrieval pattern; reject at
        # settings-construction time so the error doesn't surface deep
        # in the retriever stack.
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            KHopRetrieveSettings(k_hop=0)


class TestAtlasRagSettings:
    def test_defaults(self) -> None:
        # Defaults assume the Gemini cloud path (matches the project's
        # primary smoke configuration); switch to ollama for PCAD.
        s = AtlasRagRetrieveSettings(_env_file=None)
        assert s.provider == "openai"
        assert s.model_id == "gemini-2.5-flash"
        assert s.api_key_env == "GEMINI_API_KEY"
        assert "generativelanguage.googleapis.com" in (s.base_url or "")
        assert s.keyword == "transcriptions.json"
        assert s.include_events is True
        assert s.include_concept is True

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_ATLAS_RAG_PROVIDER", "ollama")
        monkeypatch.setenv("ARANDU_ATLAS_RAG_MODEL_ID", "qwen3:14b")
        monkeypatch.setenv("ARANDU_ATLAS_RAG_BASE_URL", "http://localhost:11434/v1")
        s = AtlasRagRetrieveSettings()
        assert s.provider == "ollama"
        assert s.model_id == "qwen3:14b"
        assert s.base_url == "http://localhost:11434/v1"

    def test_provider_normalized_to_lowercase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Without normalization, ARANDU_ATLAS_RAG_PROVIDER=Ollama would
        # later break the LLMProvider(value) coercion (enum values are
        # lowercase). Settings flatten the case here so the dispatch is
        # robust to env-var case.
        monkeypatch.setenv("ARANDU_ATLAS_RAG_PROVIDER", "Ollama")
        assert AtlasRagRetrieveSettings().provider == "ollama"

    def test_base_url_stays_none_for_ollama_without_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Bug fix from PR #108 review: previously ``base_url`` defaulted
        # to Gemini's URL, so ``ARANDU_ATLAS_RAG_PROVIDER=ollama`` would
        # silently send Ollama requests to Gemini's endpoint. The per-
        # provider validator now leaves ``base_url`` None for ollama so
        # ``LLMClient.PROVIDER_URLS[OLLAMA]`` (localhost:11434) wins.
        monkeypatch.delenv("ARANDU_ATLAS_RAG_BASE_URL", raising=False)
        monkeypatch.setenv("ARANDU_ATLAS_RAG_PROVIDER", "ollama")
        assert AtlasRagRetrieveSettings().base_url is None

    def test_base_url_defaults_to_gemini_for_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARANDU_ATLAS_RAG_BASE_URL", raising=False)
        monkeypatch.setenv("ARANDU_ATLAS_RAG_PROVIDER", "openai")
        s = AtlasRagRetrieveSettings()
        assert s.base_url is not None
        assert "generativelanguage.googleapis.com" in s.base_url

    def test_explicit_base_url_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_ATLAS_RAG_PROVIDER", "openai")
        monkeypatch.setenv("ARANDU_ATLAS_RAG_BASE_URL", "https://custom.example/v1/")
        s = AtlasRagRetrieveSettings()
        assert s.base_url == "https://custom.example/v1/"


def test_khop_top_k_seeds_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from arandu.shared.rag.retrieve.settings import KHopRetrieveSettings

    assert KHopRetrieveSettings().top_k_seeds == 50
    monkeypatch.setenv("ARANDU_KHOP_TOP_K_SEEDS", "25")
    assert KHopRetrieveSettings().top_k_seeds == 25
    assert not hasattr(KHopRetrieveSettings(), "max_postings")
