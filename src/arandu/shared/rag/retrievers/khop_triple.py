"""K-hop triple-injection retriever — preserves methodology §6.4 paradigm.

Sibling of :class:`KHopSubgraphRetriever`. Same entity-link + k-hop machinery,
but emits **linearized triples** instead of passage references — the
``[type1] head_label --[relation]--> [type2] tail_label`` form described
in ``docs/methodology.md §6.4`` Stage 2. The methodology's pre-Phase-C
KC algorithm linearized the subgraph as triples and fed them to the
Answerer as the sole context; this retriever brings that paradigm back
as a **fifth Phase C arm** so the ``passage-NetworkX vs
triple-NetworkX`` delta becomes a deliberate methodological finding.

Why a separate retriever (not an Answerer context-format strategy):

Spec §5.1 makes "Answerer held CONSTANT across arms" the methodological
cornerstone. Pushing triple-vs-passage into the Answerer would add a
knob on the supposedly-constant Answerer, couple it to the KG, and
break the apples-to-apples cross-arm comparison. Making it a sibling
retriever preserves Answerer constancy: each arm is just "a different
retriever, same Answerer."

Triple delivery rides on the new ``RetrievedPassage.payload`` field
(see ``shared/rag/schemas.py``). When set, the Answerer uses payload
verbatim as the prompt context for that record, bypassing the standard
offset-based source-text resolution.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import networkx as nx

from arandu.shared.rag.schemas import RetrievedPassage

if TYPE_CHECKING:
    from collections.abc import Iterable


# The shared helpers (tokenizer, stopwords, postings cap) are duplicated from
# :mod:`arandu.shared.rag.retrievers.khop_subgraph` rather than imported. Reason:
# the constants are private (`_TOKEN_RE`, `_STOPWORDS`, etc.) and the user
# may want them to evolve independently. A follow-up refactor can extract
# them into a sibling helper module once both retrievers stabilise.
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
# Concept nodes ARE linkable (a question may name a concept directly), but
# are EXCLUDED from triple endpoints below — see `_TRIPLE_ENDPOINT_TYPES`.
_LINKABLE_TYPES: frozenset[str] = frozenset({"entity", "event", "concept"})
# Triple endpoints deliberately exclude `concept` nodes. atlas-rag's
# conceptualization stage (AutoSchemaKG schema induction, see
# `kg/atlas_backend.py` + methodology §5.3) attaches every entity to its
# concept(s) via `has_concept` edges — a star pattern where the concept
# node accumulates extremely high in-degree. With endpoint-degree
# scoring, those `entity --[has_concept]--> concept` edges dominate any
# question's top-k results because the concept is a hub, not because
# the relation is semantically relevant.
#
# Methodology §6.4 Stage 2's example triple was
# ``[PERSON] Maria --[VIVE_EM]--> [LOCATION] Barra`` — a semantic relation
# between entities, not a type-attachment. Excluding concept endpoints
# restores that paradigm. The first smoke against `test-kg-04`
# (2026-05-23) without this exclusion returned identical top-3
# `has_concept` triples for every question.
_TRIPLE_ENDPOINT_TYPES: frozenset[str] = frozenset({"entity", "event"})
_MIN_TOKEN_LEN = 3
_DEFAULT_MAX_POSTINGS = 200
_INFINITY = 10**9  # sentinel for "unreachable from any seed" in distance dicts

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "o", "as", "os", "um", "uma", "uns", "umas",
        "de", "do", "da", "dos", "das", "no", "na", "nos", "nas",
        "em", "por", "para", "com", "sem", "sob", "sobre", "ate", "até",
        "e", "ou", "mas", "porem", "porém", "que", "se", "como", "quando",
        "onde", "qual", "quais", "quem", "porque", "porquê",
        "eu", "tu", "ele", "ela", "nós", "vos", "vós", "eles", "elas",
        "meu", "minha", "teu", "tua", "seu", "sua", "nosso", "nossa",
        "este", "esta", "esse", "essa", "aquele", "aquela", "isto", "isso", "aquilo",
        "ser", "ter", "estar", "haver", "ir", "vir", "fazer",
        "é", "foi", "era", "são", "está", "estão", "tem", "têm", "ha", "há",
        "muito", "muitos", "muita", "muitas", "pouco", "poucos", "todo", "todos",
        "não", "nao", "sim", "já", "ja", "ainda", "também", "tambem",
        "an", "the", "of", "in", "on", "at", "to", "for", "with",
        "by", "from", "into", "about", "and", "or", "but",
        "is", "are", "was", "were", "be", "been", "being",
        "i", "you", "he", "she", "it", "we", "they", "this", "that",
        "does", "did", "have", "has", "had",
        "not", "yes", "what", "which", "who", "whom", "when", "where", "why", "how",
    }
)  # fmt: skip


def _tokenize(text: str, *, filter_stopwords: bool = False) -> list[str]:
    """Whitespace + punctuation split, NFKC-normalised, casefolded.

    Mirrors :func:`arandu.shared.rag.retrievers.khop_subgraph._tokenize` so the
    entity-link stage produces identical seeds across both retrievers.
    """
    normalised = unicodedata.normalize("NFKC", text).casefold()
    tokens = _TOKEN_RE.findall(normalised)
    if filter_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) >= _MIN_TOKEN_LEN]
    return tokens


def _format_triple(
    head_id: str,
    head_attrs: dict,
    relation: str,
    tail_id: str,
    tail_attrs: dict,
) -> str:
    """Render ``[type] label --[relation]--> [type] label`` for a single edge.

    Follows the format shown in ``docs/methodology.md §6.4`` Stage 2.
    Falls back to the node ID when the ``id`` attribute is missing
    (defensive — shouldn't happen on atlas-rag KGs).
    """
    head_label = head_attrs.get("id") or head_id
    head_type = head_attrs.get("type", "?")
    tail_label = tail_attrs.get("id") or tail_id
    tail_type = tail_attrs.get("type", "?")
    return f"[{head_type}] {head_label} --[{relation}]--> [{tail_type}] {tail_label}"


class KHopTripleRetriever:
    """Triple-injection variant of the NetworkX subgraph retriever.

    Same entity-link + k-hop seeding as :class:`KHopSubgraphRetriever`, but
    emits linearized triples (``[type] head --[rel]--> [type] tail``) via
    ``RetrievedPassage.payload`` rather than passage references.

    Attributes:
        retriever_id: Stable identifier; defaults to ``khop_triple``.
    """

    RETRIEVER_FAMILY = "khop"
    DEFAULT_RETRIEVER_ID = "khop_triple"

    retriever_id: str

    def __init__(
        self,
        kg_outputs_dir: Path,
        keyword: str = "transcriptions.json",
        k_hop: int = 2,
        max_postings: int = _DEFAULT_MAX_POSTINGS,
        retriever_id: str | None = None,
    ) -> None:
        """Load the KG and pre-build the entity-label index.

        Args:
            kg_outputs_dir: The atlas-rag output dir, same shape as
                :class:`KHopSubgraphRetriever` and :class:`AtlasRagRetriever`.
                Must contain ``kg_graphml/<keyword>_graph.graphml``. Unlike
                the passage retrievers, no ``kg_extraction/`` access is
                needed — this retriever operates purely on the graphml.
            keyword: atlas-rag's filename pattern.
            k_hop: Ego-graph radius. Must be ``>= 1``.
            max_postings: IDF-style threshold for the entity link
                (drops question tokens that hit more than this many
                linkable nodes). See :class:`KHopSubgraphRetriever` for the
                calibration rationale.
            retriever_id: Optional override of :attr:`DEFAULT_RETRIEVER_ID`.

        Raises:
            FileNotFoundError: If the graphml is missing.
            ValueError: If ``k_hop < 1`` or ``max_postings < 1``.
        """
        if k_hop < 1:
            raise ValueError(f"k_hop must be >= 1, got {k_hop}")
        if max_postings < 1:
            raise ValueError(f"max_postings must be >= 1, got {max_postings}")

        graphml_path = kg_outputs_dir / "kg_graphml" / f"{keyword}_graph.graphml"
        if not graphml_path.exists():
            raise FileNotFoundError(f"kg.graphml not found at {graphml_path}")

        self.retriever_id = retriever_id or self.DEFAULT_RETRIEVER_ID
        self._k_hop = k_hop
        self._max_postings = max_postings
        self._kg: nx.DiGraph = nx.read_graphml(str(graphml_path))

        # Token-level inverted index over linkable node labels.
        self._token_to_nodes: dict[str, set[str]] = defaultdict(set)
        for node_id, attrs in self._kg.nodes(data=True):
            if attrs.get("type") in _LINKABLE_TYPES:
                for token in _tokenize(attrs.get("id", "")):
                    self._token_to_nodes[token].add(node_id)

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        """Run entity-link + k-hop subgraph + triple linearization.

        Args:
            question: Natural-language query.
            top_k: Maximum number of triples to return.

        Returns:
            Ranked list of :class:`RetrievedPassage`. Each record carries
            a synthetic ``chunk_id`` (``triple:<sha1[:16]>``) and a
            populated ``payload`` field with the linearized triple.
            Empty entity-link → empty list (graph-floor guarantee).
        """
        seeds = self._entity_link(question)
        if not seeds:
            return []

        subgraph_nodes: set[str] = set()
        for seed in seeds:
            ego = nx.ego_graph(self._kg, seed, radius=self._k_hop, undirected=True)
            subgraph_nodes.update(ego.nodes)

        # Induced subgraph + undirected shortest-path distances from any
        # seed. **Scoring is seed-proximity**, not endpoint-degree.
        # Rationale: degree-based scoring rewards graph hubs (`Dona Gilda`,
        # generic locations) regardless of question relevance — the first
        # post-concept-filter smoke on `test-kg-04` returned identical
        # hub-entity triples across very different questions. Proximity
        # to the question's anchored seeds is the signal we actually want.
        induced = self._kg.subgraph(subgraph_nodes)
        induced_undir = induced.to_undirected(as_view=True)
        node_dist: dict[str, int] = {}
        for seed in seeds:
            for n, d in nx.single_source_shortest_path_length(
                induced_undir, seed, cutoff=self._k_hop
            ).items():
                if d < node_dist.get(n, _INFINITY):
                    node_dist[n] = d

        # Collect entity-entity / event-entity / event-event triples — skip:
        # - any edge touching a non-{entity,event} node (passages have 2 KB
        #   text in `id`; concepts are atlas-rag schema-induction hubs);
        # - self-loops (`head == tail`): structurally weird, methodologically
        #   uninformative — a relation from an entity to itself doesn't
        #   carry the cross-entity semantic content §6.4 calls for.
        triples: list[tuple[str, float]] = []
        seen: set[str] = set()
        for head_id, tail_id, attrs in induced.edges(data=True):
            if head_id == tail_id:
                continue
            head_attrs = induced.nodes[head_id]
            tail_attrs = induced.nodes[tail_id]
            if (
                head_attrs.get("type") not in _TRIPLE_ENDPOINT_TYPES
                or tail_attrs.get("type") not in _TRIPLE_ENDPOINT_TYPES
            ):
                continue
            relation = attrs.get("relation", "related_to")
            triple_str = _format_triple(head_id, head_attrs, relation, tail_id, tail_attrs)
            if triple_str in seen:
                continue
            seen.add(triple_str)
            # Score: inverse total distance from seeds. Both endpoints
            # touching a seed (dist 0+0) → score 1.0; one hop away each
            # → 1/3 ≈ 0.33; deeper → tail off rapidly. This ties
            # ranking back to "which edges does the question actually
            # ask about" rather than "which edges sit on graph hubs".
            dist_sum = node_dist.get(head_id, _INFINITY) + node_dist.get(tail_id, _INFINITY)
            score = 1.0 / (1.0 + dist_sum)
            triples.append((triple_str, score))

        if not triples:
            return []

        triples.sort(key=lambda t: t[1], reverse=True)
        ranked = triples[:top_k]
        if not ranked:
            # Caller passed top_k <= 0. The sibling KHopSubgraphRetriever
            # would slice to empty too — return cleanly rather than letting
            # `ranked[0]` raise IndexError.
            return []
        max_score = ranked[0][1] or 1.0  # avoid division by zero on degenerate KGs
        return [
            RetrievedPassage(
                chunk_id=f"triple:{hashlib.sha1(triple_str.encode('utf-8')).hexdigest()[:16]}",
                payload=triple_str,
                rank=rank,
                score=score / max_score,
                retriever_meta={
                    "score_method": "seed_proximity",
                    "k_hop": self._k_hop,
                },
            )
            for rank, (triple_str, score) in enumerate(ranked)
        ]

    def _entity_link(self, question: str) -> Iterable[str]:
        """Identical contract to :meth:`KHopSubgraphRetriever._entity_link`."""
        question_tokens = set(_tokenize(question, filter_stopwords=True))
        if not question_tokens:
            return []
        seeds: set[str] = set()
        for token in question_tokens:
            postings = self._token_to_nodes.get(token, ())
            if len(postings) > self._max_postings:
                continue
            seeds.update(postings)
        return seeds
