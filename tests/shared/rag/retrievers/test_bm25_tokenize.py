"""Tests for BM25 tokenizers — whitespace fallback path (spec §4.3)."""

from __future__ import annotations

from arandu.shared.rag.retrievers._bm25_tokenize import (
    _whitespace_tokenize,
    english_tokenizer,
    portuguese_tokenizer,
)


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
