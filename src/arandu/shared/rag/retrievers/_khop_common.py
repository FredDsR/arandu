"""Shared tokenizer + stopword constants for the k-hop retriever family.

Both :class:`KHopSubgraphRetriever` (passage variant) and
:class:`KHopTripleRetriever` (triple variant) run the SAME entity-link
+ k-hop machinery — they differ only in how they score and emit results
from the induced subgraph. Keeping the tokenizer + stopwords here means
both arms produce identical seed sets for the same question, which is
load-bearing for the cross-arm comparison the methodology rewrite will
report.

Private to the ``retrievers/`` package — the underscore prefix signals
"don't import this from elsewhere in the codebase." If a third k-hop
variant ever appears, it imports from here too.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_MIN_TOKEN_LEN = 3
_DEFAULT_MAX_POSTINGS = 200

# Linkable node types (used by both arms during entity-linking). Concept
# nodes ARE linkable — a question may name a concept directly — but the
# triple variant additionally excludes them from triple ENDPOINTS via its
# own constant (see ``khop_triple._TRIPLE_ENDPOINT_TYPES``).
_LINKABLE_TYPES: frozenset[str] = frozenset({"entity", "event", "concept"})

# Short, hand-curated PT + EN stopword list. Deliberately tiny (the most
# common articles, prepositions, conjunctions, copulas) rather than
# pulling in spaCy's full list — the k-hop arms are the sanity baseline,
# and a small list gives most of the signal-vs-noise improvement without
# the dependency. The first smoke against `test-kg-04` (a 14k-node KG)
# without any filtering linked common tokens like "a"/"que"/"de" to
# thousands of entities, producing identical query results across very
# different questions in ~25 min wall time per query.
_STOPWORDS: frozenset[str] = frozenset(
    {
        # Portuguese
        "a", "o", "as", "os", "um", "uma", "uns", "umas",
        "de", "do", "da", "dos", "das", "no", "na", "nos", "nas",
        "em", "por", "para", "com", "sem", "sob", "sobre", "ate", "até",
        "e", "ou", "mas", "porem", "porém", "que", "se", "como", "quando",
        "onde", "qual", "quais", "quem", "porque", "porquê",
        "eu", "tu", "ele", "ela", "nós", "vos", "vós", "eles", "elas",
        "meu", "minha", "teu", "tua", "seu", "sua", "nosso", "nossa",
        "este", "esta", "esse", "essa", "aquele", "aquela",
        "isto", "isso", "aquilo",
        "ser", "ter", "estar", "haver", "ir", "vir", "fazer",
        "é", "foi", "era", "são", "está", "estão", "tem", "têm", "ha", "há",
        "muito", "muitos", "muita", "muitas",
        "pouco", "poucos", "todo", "todos",
        "não", "nao", "sim", "já", "ja", "ainda", "também", "tambem",
        # English (queries may mix PT/EN in this corpus)
        "an", "the", "of", "in", "on", "at", "to", "for", "with",
        "by", "from", "into", "about", "and", "or", "but",
        "is", "are", "was", "were", "be", "been", "being",
        "i", "you", "he", "she", "it", "we", "they", "this", "that",
        "does", "did", "have", "has", "had",
        "not", "yes", "what", "which", "who", "whom",
        "when", "where", "why", "how",
    }
)  # fmt: skip


def _tokenize(text: str, *, filter_stopwords: bool = False) -> list[str]:
    """Whitespace + punctuation split, lowercased + NFKC-normalised.

    When ``filter_stopwords`` is true, tokens in :data:`_STOPWORDS` and
    tokens shorter than :data:`_MIN_TOKEN_LEN` are dropped. We do NOT
    filter at index-build time (node labels keep all tokens so multi-word
    names like "rio Uruguai" remain linkable when the question mentions
    only the rare half); only the QUERY tokens are filtered, so a question
    composed only of stopwords degenerates to an empty entity link and
    the graph-floor guarantee fires (returns ``[]``).
    """
    normalised = unicodedata.normalize("NFKC", text).casefold()
    tokens = _TOKEN_RE.findall(normalised)
    if filter_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) >= _MIN_TOKEN_LEN]
    return tokens
