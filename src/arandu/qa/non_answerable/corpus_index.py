"""Absence-check structures for perturbation verification (spec §7.5).

A swap is only valid if the replacement entity is absent from BOTH:

- the KG node set (:func:`load_kg_node_set`) - entities the extractor saw, and
- the source corpus (:class:`SourceCorpusIndex`) - a broader bag of NER
  spans + alpha tokens straight from the transcriptions.

Both use lower-cased, whitespace-stripped normalization so case and
spacing differences don't leak a present entity through as "absent".
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arandu.shared.schemas import EnrichedRecord

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

# Tokens shorter than this are skipped (drops articles/prepositions like
# "de", "em", "as"). Applied inclusively: a token of exactly this length
# is kept.
_MIN_ALPHA_TOKEN_LEN = 4


def _normalize(text: str) -> str:
    """Lower-case + strip; the single normalization used on both sides."""
    return text.strip().lower()


def load_kg_node_set(graphml_path: Path) -> set[str]:
    """Load the lower-cased label set from a KG GraphML file.

    Falls back to the node id when a node carries no ``label`` attribute.
    Returns an empty set (with a warning) if NetworkX or the file is
    unavailable - the corpus index still provides an absence check.

    Args:
        graphml_path: Path to ``kg_graphml/<keyword>_graph.graphml``.

    Returns:
        Set of normalized node labels.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx unavailable; KG node absence check disabled.")
        return set()
    if not graphml_path.exists():
        logger.warning("KG GraphML absent at %s; KG node absence check disabled.", graphml_path)
        return set()
    graph = nx.read_graphml(str(graphml_path))
    return {_normalize(data.get("label") or node_id) for node_id, data in graph.nodes(data=True)}


class SourceCorpusIndex:
    """Bag of named entities + alpha tokens drawn from the source corpus.

    Built once per run over every :class:`EnrichedRecord` transcription.
    Membership is the second absence gate for a candidate replacement
    entity. Uses spaCy ``pt_core_news_sm`` with NER enabled; when the
    model is unavailable it degrades to an alpha-token-only bag (logged)
    so generation still runs, just with a weaker corpus gate.
    """

    def __init__(self, transcription_dir: Path) -> None:
        """Build the index from every ``EnrichedRecord`` under ``transcription_dir``."""
        self._spans: set[str] = set()
        self._nlp = _portuguese_nlp()
        self._build(transcription_dir)

    def __contains__(self, candidate: str) -> bool:
        """True if ``candidate`` (normalized) appears anywhere in the corpus."""
        return _normalize(candidate) in self._spans

    def __len__(self) -> int:
        """Number of distinct spans indexed."""
        return len(self._spans)

    def _build(self, transcription_dir: Path) -> None:
        if not transcription_dir.exists():
            logger.warning(
                "Transcription dir absent at %s; corpus absence check will be empty.",
                transcription_dir,
            )
            return
        for path in sorted(transcription_dir.glob("*.json")):
            try:
                record = EnrichedRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Skipping unreadable transcription %s: %s", path, exc)
                continue
            self._index_text(record.transcription_text)

    def _index_text(self, text: str) -> None:
        if self._nlp is not None:
            doc = self._nlp(text)
            self._spans |= {_normalize(ent.text) for ent in doc.ents}
            self._spans |= {
                _normalize(tok.text)
                for tok in doc
                if tok.is_alpha and len(tok.text) >= _MIN_ALPHA_TOKEN_LEN
            }
            return
        # Fallback: whitespace alpha-token bag (no NER).
        self._spans |= {
            _normalize(tok)
            for tok in text.split()
            if tok.isalpha() and len(tok) >= _MIN_ALPHA_TOKEN_LEN
        }


def _portuguese_nlp() -> Callable[[str], object] | None:
    """Load spaCy ``pt_core_news_sm`` with NER; return None if unavailable."""
    try:
        import spacy
    except ImportError:
        logger.warning("spaCy unavailable; corpus index falls back to token-only bag.")
        return None
    try:
        # Keep NER; drop the parser (unused) to save time on long transcripts.
        return spacy.load("pt_core_news_sm", disable=["parser", "lemmatizer"])
    except OSError:
        logger.warning(
            "spaCy 'pt_core_news_sm' unavailable; corpus index falls back to token-only bag."
        )
        return None
