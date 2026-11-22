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

import math
import re
import threading
import unicodedata
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import networkx as nx

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_MIN_TOKEN_LEN = 3
_DEFAULT_TOP_K_SEEDS = 50

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


@lru_cache(maxsize=1)
def _lemmatizer() -> Callable[[str], list[str]] | None:
    """Lazy spaCy lemmatizer (pt_core_news_sm); None when unavailable.

    Returns:
        A callable that accepts a string and returns a list of casefolded lemmas,
        or ``None`` if spaCy or its Portuguese model is not installed.
    """
    try:
        import spacy
    except ImportError:
        return None
    try:
        nlp = spacy.load("pt_core_news_sm", disable=["ner", "parser"])
    except OSError:
        return None

    # spaCy does not formally guarantee thread safety for a shared Language
    # object (Vocab/StringStore mutate during calls). Mirror the lock pattern
    # from _bm25_tokenize._spacy_tokenizer to serialize access across workers.
    lock = threading.Lock()

    def lemmatize(text: str) -> list[str]:
        with lock:
            return [tok.lemma_.casefold() for tok in nlp(text) if not tok.is_space]

    return lemmatize


def _tokenize(text: str, *, filter_stopwords: bool = False) -> list[str]:
    """Tokenize ``text`` with spaCy lemmatization when available, else whitespace split.

    Uses ``pt_core_news_sm`` lemmatization to collapse inflectional variants so
    that query tokens match node labels regardless of number or conjugation (e.g.
    "enchentes" -> "enchente", "pescavam" -> "pescar"). Derivational variants
    (e.g. "pescador" vs "pesca") remain distinct — accepted limitation of
    lexical-only matching.

    When spaCy or ``pt_core_news_sm`` is unavailable, falls back to NFKC
    normalisation + ``\\w+`` regex split (the original whitespace path).

    When ``filter_stopwords`` is true, tokens in :data:`_STOPWORDS` and tokens
    shorter than :data:`_MIN_TOKEN_LEN` are dropped. We do NOT filter at
    index-build time (node labels keep all tokens so multi-word names like
    "rio Uruguai" remain linkable when the question mentions only the rare
    half); only the QUERY tokens are filtered, so a question composed only of
    stopwords degenerates to an empty entity link and the graph-floor guarantee
    fires (returns ``[]``).

    Args:
        text: Raw text to tokenize.
        filter_stopwords: When ``True``, drop stopwords and short tokens.

    Returns:
        List of casefolded token strings (lemmatized when spaCy is available).
    """
    lemmatize = _lemmatizer()
    if lemmatize is not None:
        tokens = lemmatize(text)
    else:
        normalised = unicodedata.normalize("NFKC", text).casefold()
        tokens = _TOKEN_RE.findall(normalised)
    if filter_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) >= _MIN_TOKEN_LEN]
    return tokens


def build_label_index(kg: nx.DiGraph) -> tuple[dict[str, set[str]], int]:
    """Build the token->nodes inverted index over linkable node labels.

    Returns ``(token_to_nodes, n_linkable)`` where ``n_linkable`` is the
    number of linkable nodes (the IDF document count — the universe the
    labels are drawn from). Labels are NOT stopword-filtered (index-build
    path) so multi-word names stay linkable on their rare half.

    Args:
        kg: A NetworkX directed graph with node attributes ``id`` (label text)
            and ``type`` (used to decide linkability via ``_LINKABLE_TYPES``).

    Returns:
        A tuple of (token_to_nodes, n_linkable) where token_to_nodes maps each
        token to the set of node IDs whose label contains it, and n_linkable is
        the count of linkable nodes.
    """
    from collections import defaultdict

    token_to_nodes: dict[str, set[str]] = defaultdict(set)
    n_linkable = 0
    for node_id, attrs in kg.nodes(data=True):
        if attrs.get("type") in _LINKABLE_TYPES:
            n_linkable += 1
            for token in _tokenize(attrs.get("id", "")):
                token_to_nodes[token].add(node_id)
    return token_to_nodes, n_linkable


def score_seeds(
    question: str, token_to_nodes: dict[str, set[str]], n_linkable: int
) -> dict[str, float]:
    """Map candidate seed node -> summed smoothed-IDF weight of its linking tokens.

    Smoothed IDF ``log((N+1)/(df+1))`` stays strictly non-negative (a token in
    every node still contributes a weight of 0.0, so the rarest available token
    can always seed). A node matched by several query tokens accumulates their
    weights.

    Args:
        question: Raw question text; query tokens are stopword-filtered.
        token_to_nodes: Inverted index from :func:`build_label_index`.
        n_linkable: Total linkable node count from :func:`build_label_index`.

    Returns:
        Dict mapping node IDs to their accumulated IDF-weight score.
    """
    from collections import defaultdict

    weights: dict[str, float] = defaultdict(float)
    for token in set(_tokenize(question, filter_stopwords=True)):
        postings = token_to_nodes.get(token)
        if not postings:
            continue
        idf = math.log((n_linkable + 1) / (len(postings) + 1))
        for node in postings:
            weights[node] += idf
    return weights


def link_entities(
    question: str,
    token_to_nodes: dict[str, set[str]],
    n_linkable: int,
    *,
    top_k_seeds: int = _DEFAULT_TOP_K_SEEDS,
) -> list[str]:
    """Entity-link a question to the top-K KG seed nodes by IDF weight.

    Keeps every matched token (no hard drop); weights candidate seeds by
    summed smoothed IDF; returns the top-K by weight. Empty only when no
    query token matches any label (true graph-floor). The top-K budget
    bounds downstream ego-graph size (the role previously played by the
    per-token posting cap).

    Args:
        question: Raw question text used to identify seed nodes.
        token_to_nodes: Inverted index from :func:`build_label_index`.
        n_linkable: Total linkable node count from :func:`build_label_index`.
        top_k_seeds: Maximum number of seed nodes to return.

    Returns:
        List of node IDs (strings), ranked by descending IDF weight, capped
        at ``top_k_seeds``. Returns ``[]`` when no query token matches any label.
    """
    weights = score_seeds(question, token_to_nodes, n_linkable)
    if not weights:
        return []
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return [node for node, _ in ranked[:top_k_seeds]]


def subgraph_node_distances(kg: nx.DiGraph, seeds: Iterable[str], k_hop: int) -> dict[str, int]:
    """Min undirected hop-distance from any seed to each node within ``k_hop``.

    Unions the ``k_hop`` ego graph of each seed, induces that node set, and runs
    single-source shortest paths (undirected, cutoff ``k_hop``) from every seed,
    keeping the minimum distance per reached node. Seeds map to 0. The returned
    dict's keys are exactly the in-``k_hop`` subgraph node set, so callers that
    also need that subgraph can use ``kg.subgraph(dist.keys())`` without a second
    ego-graph pass.

    Args:
        kg: The knowledge graph.
        seeds: Entity-linked seed node IDs.
        k_hop: Ego-graph radius / shortest-path cutoff.

    Returns:
        ``{node_id: min_distance}`` for every node within ``k_hop`` of a seed;
        empty when ``seeds`` is empty.
    """
    import networkx as nx

    seeds = list(seeds)
    if not seeds:
        return {}
    subgraph_nodes: set[str] = set()
    for seed in seeds:
        subgraph_nodes.update(nx.ego_graph(kg, seed, radius=k_hop, undirected=True).nodes)
    induced_undir = kg.subgraph(subgraph_nodes).to_undirected(as_view=True)
    dist: dict[str, int] = {}
    for seed in seeds:
        for node, d in nx.single_source_shortest_path_length(
            induced_undir, seed, cutoff=k_hop
        ).items():
            if node not in dist or d < dist[node]:
                dist[node] = d
    return dist
