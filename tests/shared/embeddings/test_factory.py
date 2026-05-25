"""Tests for ``shared/embeddings/factory.py`` — settings + provider dispatch."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from arandu.shared.embeddings import EmbedderSettings, GeminiEmbedder, build_embedder


class TestEmbedderSettings:
    def test_defaults(self) -> None:
        # Thesis runs are Gemini-driven by default. The 001 model is the
        # one Google currently supports (004 was removed in 2025-Q1).
        s = EmbedderSettings(_env_file=None)
        assert s.provider == "gemini"
        assert s.model == "gemini-embedding-001"
        assert s.api_key_env == "GEMINI_API_KEY"
        # max_retries=8 is calibrated against the test-kg-04 smoke; the SDK
        # default (2) is too low for 75k-embedding builds under Gemini's quota.
        assert s.max_retries == 8

    def test_provider_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_EMBEDDER_PROVIDER", "sentence_transformers")
        monkeypatch.setenv("ARANDU_EMBEDDER_MODEL", "intfloat/multilingual-e5-large-instruct")
        s = EmbedderSettings()
        assert s.provider == "sentence_transformers"
        assert s.model == "intfloat/multilingual-e5-large-instruct"


class TestBuildEmbedder:
    def test_gemini_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        settings = EmbedderSettings(provider="gemini", model="m", api_key_env="GEMINI_API_KEY")

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            build_embedder(settings)

    def test_gemini_returns_gemini_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        settings = EmbedderSettings(provider="gemini", model="gemini-embedding-001")

        with patch("openai.OpenAI") as mock_openai:
            embedder = build_embedder(settings)

        assert isinstance(embedder, GeminiEmbedder)
        assert embedder.model == "gemini-embedding-001"
        # The OpenAI client must be pointed at Gemini's compatibility URL,
        # not OpenAI proper. Locking the URL prevents accidental drift to
        # the wrong backend.
        mock_openai.assert_called_once()
        _, kwargs = mock_openai.call_args
        assert kwargs["api_key"] == "test-key"
        assert "generativelanguage.googleapis.com" in kwargs["base_url"]
        # max_retries is forwarded so large embedding builds survive 429s
        # under Gemini's quota (the SDK's default of 2 is too low for
        # 75k-embedding runs — the smoke script uses 8).
        assert kwargs["max_retries"] == 8

    def test_max_retries_override_via_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        settings = EmbedderSettings(provider="gemini", model="m", max_retries=15)

        with patch("openai.OpenAI") as mock_openai:
            build_embedder(settings)

        assert mock_openai.call_args.kwargs["max_retries"] == 15

    def test_sentence_transformers_dispatches_to_atlas_rag_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The atlas-rag SentenceTransformerEmbeddingModel constructor pulls
        # weights; patch the class so the test stays import-only.
        # Skip if atlas_rag isn't installed (CI runs without --extra kg).
        pytest.importorskip("atlas_rag.vectorstore.embedding_model")
        settings = EmbedderSettings(provider="sentence_transformers", model="some/local-model")

        with patch(
            "atlas_rag.vectorstore.embedding_model.SentenceTransformerEmbeddingModel"
        ) as mock_cls:
            mock_cls.return_value = "embedder-instance"
            result = build_embedder(settings)

        mock_cls.assert_called_once_with("some/local-model")
        assert result == "embedder-instance"

    def test_unknown_provider_raises_valueerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Defensive: provider is a Literal, but env-var-driven configs
        # can still feed an arbitrary string into the field. Pydantic
        # validation catches this at construction; our factory's else
        # branch is a belt-and-braces guard.
        settings = EmbedderSettings.model_construct(
            provider="nonexistent",  # type: ignore[arg-type]
            model="m",
            api_key_env="GEMINI_API_KEY",
        )
        with pytest.raises(ValueError, match="Unknown embedder provider"):
            build_embedder(settings)
