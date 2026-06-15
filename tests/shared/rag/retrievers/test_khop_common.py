from __future__ import annotations

import pytest

from arandu.shared.rag.retrievers import _khop_common as kc


class TestTokenize:
    def test_lemmatizes_inflectional_variants(self) -> None:
        pytest.importorskip("spacy")
        toks = kc._tokenize("enchentes pescavam", filter_stopwords=True)
        assert "enchente" in toks
        assert any(t.startswith("pesc") for t in toks)

    def test_filter_stopwords_drops_particles_and_short(self) -> None:
        toks = kc._tokenize("a de Barra", filter_stopwords=True)
        assert "a" not in toks and "de" not in toks
        assert "barra" in toks

    def test_no_filter_keeps_all_label_tokens(self) -> None:
        toks = kc._tokenize("rio Uruguai", filter_stopwords=False)
        assert "rio" in toks and "uruguai" in toks
