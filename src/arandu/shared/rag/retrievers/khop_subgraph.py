"""K-hop subgraph retriever — graph sanity baseline (Phase C spec §4.5).

Joel's framing for Phase C: this arm exists to disentangle "graph quality"
from "retrieval-tool quality." It uses the SAME KG as :class:`AtlasRagRetriever`
but a deliberately simpler retrieval algorithm — exact + token-overlap entity
linking against node labels, a k-hop ego subgraph walk, and passage-mention
frequency scoring. The atlas-rag minus k-hop delta is then an estimate of
atlas-rag's tool-quality contribution holding KG constant.

The "k-hop" family also covers the triple-emitting sibling
(:class:`KHopTripleRetriever`); both share the entity-link + k-hop machinery
and differ only in what they return to the Answerer (passages vs linearized
triples).

Implementation notes:

- Pure-python, no atlas-rag dependency. Loads the atlas-rag GraphML
  (``kg_graphml/<keyword>_graph.graphml`` under ``kg_outputs_dir``) directly
  via ``networkx.read_graphml`` — works without the ``--extra kg`` install.
- The retriever walks an UNDIRECTED ego graph (`nx.ego_graph(..., undirected=True)`)
  from the entity-linked seed nodes. The KG itself is a directed atlas-rag
  graph; direction is dropped because mentions / relations both flow in
  meaningful ways for our scoring purpose.
- Returned ``chunk_id`` is the atlas-rag synthesized passage_id
  (``<source_file_id>:<chunk_index>``) — same namespace as the
  ``passage_offsets.json`` sidecar from PR #100, so downstream judges
  that consult offsets can join directly. The bridge from KG passage
  node text to that ID is built once at construction via
  :func:`arandu.kg.passage_offsets.build_passage_text_to_atlas_passage_id`.
  Passages whose KG node text has no matching ``kg_extraction`` record
  (rare; happens if the JSONL was pruned or the KG was rebuilt against
  a different corpus snapshot) are dropped from results.
- Empty entity-link (no question token overlaps any node label) → returns
  ``[]`` cleanly. Joel's "graph-floor" guarantee.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import networkx as nx

from arandu.kg.passage_offsets import (
    build_passage_text_to_atlas_passage_id,
    strip_atlas_header,
)
from arandu.shared.rag.retrievers._khop_common import (
    _DEFAULT_TOP_K_SEEDS,
    build_label_index,
    link_entities,
)
from arandu.shared.rag.schemas import RetrievedPassage

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


class KHopSubgraphRetriever:
    """K-hop subgraph retriever over an atlas-rag-built KG.

    Attributes:
        retriever_id: Stable identifier; defaults to ``khop_passage``.
    """

    RETRIEVER_FAMILY = "khop"
    DEFAULT_RETRIEVER_ID = "khop_passage"

    retriever_id: str

    def __init__(
        self,
        kg_outputs_dir: Path,
        keyword: str = "transcriptions.json",
        k_hop: int = 2,
        top_k_seeds: int = _DEFAULT_TOP_K_SEEDS,
        retriever_id: str | None = None,
    ) -> None:
        """Load the KG, the ``passage_text → passage_id`` map, and the entity-label index.

        Args:
            kg_outputs_dir: The atlas-rag output dir, typically
                ``results/<pipeline_id>/kg/outputs/atlas_output/``. Must
                contain ``kg_graphml/<keyword>_graph.graphml`` and
                ``kg_extraction/*.json``. Same layout the
                :class:`AtlasRagRetriever` expects.
            keyword: atlas-rag's filename pattern; defaults to the project
                convention ``"transcriptions.json"`` (cf.
                :mod:`arandu.kg.atlas_backend`).
            k_hop: Ego-graph radius used at retrieval time. Must be ``>= 1``.
            top_k_seeds: Max entity-link seeds kept per question (ranked by
                summed smoothed-IDF weight); bounds ego-graph size. A smaller
                value prunes low-weight seeds that would otherwise expand the
                subgraph with loosely related nodes.
            retriever_id: Optional override of :attr:`DEFAULT_RETRIEVER_ID`.

        Raises:
            FileNotFoundError: If the graphml or the ``kg_extraction/``
                directory is missing under ``kg_outputs_dir``.
            ValueError: If ``k_hop < 1`` or ``top_k_seeds < 1``.
        """
        if k_hop < 1:
            raise ValueError(f"k_hop must be >= 1, got {k_hop}")
        if top_k_seeds < 1:
            raise ValueError(f"top_k_seeds must be >= 1, got {top_k_seeds}")

        graphml_path = kg_outputs_dir / "kg_graphml" / f"{keyword}_graph.graphml"
        if not graphml_path.exists():
            raise FileNotFoundError(f"atlas-rag GraphML not found at {graphml_path}")
        kg_extraction_dir = kg_outputs_dir / "kg_extraction"
        if not kg_extraction_dir.exists():
            raise FileNotFoundError(
                f"kg_extraction dir not found at {kg_extraction_dir}. "
                f"Required to bridge KG passage nodes to atlas-rag's "
                f"synthesized passage_id namespace."
            )

        self.retriever_id = retriever_id or self.DEFAULT_RETRIEVER_ID
        self._k_hop = k_hop
        self._top_k_seeds = top_k_seeds
        self._kg: nx.DiGraph = nx.read_graphml(str(graphml_path))

        # passage_text → "<source_file_id>:<chunk_index>" — built from the
        # same JSONL iterator that powers `arandu kg-link-passages`, so the
        # IDs we hand out are byte-identical to the ones in
        # `passage_offsets.json`. The downstream `passage_coverage`
        # offset-variant judge consults that sidecar; sharing the
        # namespace here is what lets it find our results.
        self._text_to_passage_id: dict[str, str] = build_passage_text_to_atlas_passage_id(
            kg_extraction_dir
        )

        # Shared IDF-weighted inverted index (build_label_index covers
        # linkable nodes only) + passage-text map (iterated once here).
        self._token_to_nodes, self._n_linkable = build_label_index(self._kg)
        self._passage_text: dict[str, str] = {
            node_id: attrs.get("id", "")
            for node_id, attrs in self._kg.nodes(data=True)
            if attrs.get("type") == "passage"
        }

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        """Run the entity-link + k-hop + frequency-scoring pipeline.

        Args:
            question: Natural-language query.
            top_k: Maximum number of passages to return.

        Returns:
            A list of :class:`RetrievedPassage` with consecutive
            zero-indexed ranks and monotonically non-increasing scores.
            Empty entity link → empty list (graph-floor guarantee).
        """
        if top_k <= 0:
            # Contract: caller may pass top_k=0 to disable an arm without
            # branching elsewhere. Return [] cleanly rather than letting
            # the build-then-cap loop yield a stray record.
            return []
        seeds = self._entity_link(question)
        if not seeds:
            return []

        # Union the k-hop ego graph of each seed. NetworkX's `ego_graph`
        # returns a copy already; we union node IDs only.
        subgraph_nodes: set[str] = set()
        for seed in seeds:
            ego = nx.ego_graph(self._kg, seed, radius=self._k_hop, undirected=True)
            subgraph_nodes.update(ego.nodes)

        # Passage scoring: a passage counts only if its NODE is in the
        # subgraph (i.e. reachable from a seed within k_hop edges). The
        # score is how many other subgraph nodes reference it via
        # `file_id` (atlas-rag's mention-attribution convention; cf.
        # `kg/atlas_backend.py`). Limiting "candidate passages" to those
        # actually in the subgraph is what makes the `k_hop` knob
        # meaningful — without it, any seed entity would surface its own
        # passage regardless of edge connectivity, defeating the
        # "graph-quality vs retrieval-tool" decomposition this baseline
        # exists to test.
        candidate_passages = {n for n in subgraph_nodes if n in self._passage_text}
        passage_counts: Counter[str] = Counter()
        for node_id in subgraph_nodes:
            if node_id in candidate_passages:
                # Don't count a passage's own self-mention.
                continue
            file_id_attr = self._kg.nodes[node_id].get("file_id", "")
            if not file_id_attr:
                continue
            for pid in (p.strip() for p in file_id_attr.split(",")):
                if pid and pid in candidate_passages:
                    passage_counts[pid] += 1

        if not passage_counts:
            return []

        # Convert KG passage-node hashes to atlas-rag synthesized passage_ids
        # (``<source_file_id>:<chunk_index>`` — same namespace as
        # `passage_offsets.json` from PR #100). KG passage nodes that have no
        # matching `kg_extraction` JSONL record are dropped — they're either
        # corpus-snapshot drift or atlas-rag-internal nodes without a
        # downstream identity, and surfacing them in `RetrievedPassage`
        # would carry an opaque hash that can't be joined with the offset
        # sidecar the judges consult.
        ranked: list[tuple[str, int, str]] = []
        for passage_node_id, count in passage_counts.most_common():
            text = self._passage_text[passage_node_id]
            atlas_passage_id = self._text_to_passage_id.get(text)
            if atlas_passage_id is None:
                logger.debug(
                    "Skipping KG passage node %s — no matching kg_extraction "
                    "record (text-equality miss).",
                    passage_node_id,
                )
                continue
            ranked.append((atlas_passage_id, count, text))
            if len(ranked) >= top_k:
                break

        if not ranked:
            return []

        max_count = ranked[0][1]
        # Carry the passage text inline in `payload` (marked prose) so the
        # Answerer doesn't re-resolve `chunk_id` through `passage_offsets.json`
        # at answer time — the retriever already holds the exact text.
        # `strip_atlas_header` mirrors the offset-resolution path (which strips
        # the same `[Contexto…][Transcrição]` header before slicing), so the
        # payload matches the header-free text every other arm feeds — keeping
        # the answerer prompt and source_recovery comparable across arms. The
        # `chunk_id` stays the stable sidecar-joinable `<file_id>:<chunk_index>`,
        # and `payload_is_prose=True` keeps source_recovery's token-containment
        # lens applicable (unlike triple payloads). NB: the header-bearing raw
        # `text` is still the lookup key for `_text_to_passage_id` above; only
        # the surfaced payload is stripped.
        return [
            RetrievedPassage(
                chunk_id=atlas_passage_id,
                rank=rank,
                score=count / max_count,
                payload=strip_atlas_header(text),
                payload_is_prose=True,
                retriever_meta={
                    "score_method": "node_freq_khop",
                    "k_hop": self._k_hop,
                    "top_k_seeds": self._top_k_seeds,
                },
            )
            for rank, (atlas_passage_id, count, text) in enumerate(ranked)
        ]

    def _entity_link(self, question: str) -> Iterable[str]:
        """IDF-weighted top-K lexical entity link (see _khop_common.link_entities)."""
        return link_entities(
            question, self._token_to_nodes, self._n_linkable, top_k_seeds=self._top_k_seeds
        )
