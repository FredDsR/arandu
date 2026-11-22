"""Tests for BM25 tokenizers — whitespace fallback + spaCy path (spec §4.3)."""

from __future__ import annotations

import pytest

from arandu.shared.rag.retrievers._bm25_tokenize import (
    _whitespace_tokenize,
    english_tokenizer,
    portuguese_tokenizer,
)


def _spacy_pt_available() -> bool:
    """Return True iff spaCy and pt_core_news_sm are importable+loadable."""
    try:
        import spacy
    except ImportError:
        return False
    try:
        spacy.load("pt_core_news_sm")
    except OSError:
        return False
    return True


def _spacy_en_available() -> bool:
    """Return True iff spaCy and en_core_web_sm are importable+loadable."""
    try:
        import spacy
    except ImportError:
        return False
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        return False
    return True


class TestWhitespaceFallback:
    def test_lowercases_input(self) -> None:
        assert _whitespace_tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        assert _whitespace_tokenize("Hello, world! How's it?") == [
            "hello",
            "world",
            "how",
            "s",
            "it",
        ]

    def test_preserves_numerics(self) -> None:
        # Spec §4.3: "Numerics preserved".
        assert _whitespace_tokenize("Em 2024 caíram 350mm") == [
            "em",
            "2024",
            "caíram",
            "350mm",
        ]

    def test_preserves_portuguese_accents(self) -> None:
        # NFKC keeps composed accents — important for accent-sensitive matching.
        assert _whitespace_tokenize("Ação coração") == ["ação", "coração"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert _whitespace_tokenize("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert _whitespace_tokenize("   \n\t  ") == []

    def test_collapses_internal_whitespace(self) -> None:
        assert _whitespace_tokenize("a   b\n\tc") == ["a", "b", "c"]

    def test_unicode_nfkc_normalization(self) -> None:
        # Compatibility forms (ligature 'ﬁ' → "fi") get normalized before tokenization.
        # Using a known NFKC compatibility decomposition.
        assert _whitespace_tokenize("ﬁnal") == ["final"]


class TestTokenizerFactories:
    def test_portuguese_tokenizer_returns_callable(self) -> None:
        tok = portuguese_tokenizer()
        assert callable(tok)
        # Smoke: the returned tokenizer produces a non-empty list for typical input
        result = tok("Em que ano ocorreu a enchente?")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_english_tokenizer_returns_callable(self) -> None:
        tok = english_tokenizer()
        assert callable(tok)
        result = tok("When did the flood happen?")
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)


class TestSpacyPath:
    """Positive tests for the spaCy primary path.

    These verify that when spaCy + the models are installed, the factories
    return a lemmatizing-and-stopword-filtering tokenizer (not the fallback).
    Skipped when spaCy or a model is unavailable so the suite still passes
    on a fallback-only install.
    """

    @pytest.mark.skipif(not _spacy_pt_available(), reason="pt_core_news_sm not installed")
    def test_portuguese_spacy_lemmatizes_verbs(self) -> None:
        # "alagou" (3rd person singular preterite) → "alagar" (lemma) only via spaCy.
        # The whitespace fallback would return ["alagou"].
        tokens = portuguese_tokenizer()("A enchente alagou Itaqui.")
        assert "alagar" in tokens, f"expected lemma 'alagar' in {tokens}"

    @pytest.mark.skipif(not _spacy_pt_available(), reason="pt_core_news_sm not installed")
    def test_portuguese_spacy_drops_stopwords(self) -> None:
        # "a", "de", "e" are Portuguese stopwords — present in raw text, absent after spaCy.
        # The whitespace fallback keeps them.
        tokens = portuguese_tokenizer()("A enchente de 2024 e a chuva")
        assert "a" not in tokens
        assert "de" not in tokens
        assert "e" not in tokens
        assert "enchente" in tokens

    @pytest.mark.skipif(not _spacy_en_available(), reason="en_core_web_sm not installed")
    def test_english_spacy_lemmatizes_verbs(self) -> None:
        # "flooded" → "flood" via spaCy lemmatization.
        tokens = english_tokenizer()("The river flooded the village.")
        assert "flood" in tokens, f"expected lemma 'flood' in {tokens}"


class TestTokenizerThreadSafety:
    def test_spacy_tokenizer_serializes_concurrent_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The shared spaCy Language object is not formally thread-safe; the
        # tokenizer closure must serialize access. The fake nlp records
        # concurrent entries - without a lock, two pool threads overlap.
        import sys
        import threading
        import time
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from arandu.shared.rag.retrievers import _bm25_tokenize

        in_flight = {"now": 0, "max": 0}
        gauge = threading.Lock()

        def fake_nlp(text: str) -> list[SimpleNamespace]:
            with gauge:
                in_flight["now"] += 1
                in_flight["max"] = max(in_flight["max"], in_flight["now"])
            time.sleep(0.02)
            with gauge:
                in_flight["now"] -= 1
            return [SimpleNamespace(lemma_="tok", is_stop=False, is_punct=False, is_space=False)]

        fake_spacy = MagicMock()
        fake_spacy.load.return_value = fake_nlp
        monkeypatch.setitem(sys.modules, "spacy", fake_spacy)

        tokenize = _bm25_tokenize._spacy_tokenizer("pt_core_news_sm")
        assert tokenize is not None

        threads = [threading.Thread(target=tokenize, args=("um texto",)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert in_flight["max"] == 1
