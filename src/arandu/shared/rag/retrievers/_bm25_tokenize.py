"""BM25 tokenizers — spaCy primary path with whitespace fallback (spec §4.3).

Portuguese: ``pt_core_news_sm`` (lemmatization + stopword removal). English:
``en_core_web_sm``. When spaCy or its model is unavailable, falls back to
``_whitespace_tokenize``: NFKC + lowercase + punctuation-stripping + split.
The fallback emits a logged warning at factory time.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _whitespace_tokenize(text: str) -> list[str]:
    """Tokenize via NFKC + lowercase + punctuation-strip + whitespace split.

    Numerics and Portuguese accents are preserved (NFKC keeps composed forms).
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in normalized)
    return [tok for tok in cleaned.split() if tok]


def _spacy_tokenizer(model: str) -> Callable[[str], list[str]] | None:
    """Build a spaCy lemma+stopword tokenizer; return None if spaCy/model missing."""
    try:
        import spacy
    except ImportError:
        return None
    try:
        nlp = spacy.load(model, disable=["ner", "parser"])
    except OSError:
        return None

    def tokenize(text: str) -> list[str]:
        return [
            tok.lemma_.lower()
            for tok in nlp(text)
            if not tok.is_stop and not tok.is_punct and not tok.is_space
        ]

    return tokenize


def portuguese_tokenizer() -> Callable[[str], list[str]]:
    """Return a Portuguese tokenizer (spaCy ``pt_core_news_sm`` or fallback)."""
    tokenizer = _spacy_tokenizer("pt_core_news_sm")
    if tokenizer is None:
        logger.warning("spaCy 'pt_core_news_sm' unavailable — using whitespace fallback tokenizer.")
        return _whitespace_tokenize
    return tokenizer


def english_tokenizer() -> Callable[[str], list[str]]:
    """Return an English tokenizer (spaCy ``en_core_web_sm`` or fallback)."""
    tokenizer = _spacy_tokenizer("en_core_web_sm")
    if tokenizer is None:
        logger.warning("spaCy 'en_core_web_sm' unavailable — using whitespace fallback tokenizer.")
        return _whitespace_tokenize
    return tokenizer
