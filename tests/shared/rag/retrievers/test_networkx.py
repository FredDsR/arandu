"""Tests for ``shared/rag/retrievers/networkx.py`` — graph sanity baseline (spec §4.5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import pytest

from arandu.shared.rag.protocol import Retriever
from arandu.shared.rag.retrievers.networkx import NetworkXRetriever

if TYPE_CHECKING:
    from pathlib import Path


# -- fixtures ------------------------------------------------------------


def _build_minimal_kg() -> nx.DiGraph:
    """Construct a small synthetic atlas-rag-shaped KG for retrieval tests.

    Topology:

        entity "rio" ──┐
                       ├──> passage "src_a"
        entity "enchente" ──┘
        entity "barragem" ──> passage "src_b"
        entity "isolated" (in a disconnected component, no passage)
        passage "src_a" (text about flood/river)
        passage "src_b" (text about dam)
        passage "src_c" (orphan; not referenced by any indexed entity)

    Edges go entity → passage to mimic the atlas-rag direction; the retriever
    uses an undirected ego walk so direction doesn't matter for the smoke.
    """
    kg = nx.DiGraph()
    kg.add_node("p_a", type="passage", id="A enchente do rio Uruguai em 2024.", file_id="p_a")
    kg.add_node("p_b", type="passage", id="A barragem foi reconstruída.", file_id="p_b")
    kg.add_node("p_c", type="passage", id="Texto solitário irrelevante.", file_id="p_c")
    kg.add_node("e_rio", type="entity", id="rio Uruguai", file_id="p_a")
    kg.add_node("e_enchente", type="entity", id="enchente", file_id="p_a")
    kg.add_node("e_barragem", type="entity", id="barragem", file_id="p_b")
    kg.add_node("e_isolated", type="entity", id="banhado", file_id="p_a")
    kg.add_edge("e_rio", "p_a", relation="mentions")
    kg.add_edge("e_enchente", "p_a", relation="mentions")
    kg.add_edge("e_barragem", "p_b", relation="mentions")
    # e_isolated is NOT edge-connected to any passage in the k-hop neighborhood
    # of the others — used to verify k-hop containment.
    return kg


def _write_graphml(kg: nx.DiGraph, tmp_path: Path) -> Path:
    path = tmp_path / "kg.graphml"
    nx.write_graphml(kg, path)
    return path


# -- naming + Protocol --------------------------------------------------


class TestNetworkXRetrieverId:
    """`retriever_id` follows the family-prefix convention from BM25 + atlas-rag."""

    def test_default_id(self) -> None:
        assert NetworkXRetriever.RETRIEVER_FAMILY == "networkx"
        assert NetworkXRetriever.DEFAULT_RETRIEVER_ID == "networkx_khop"


class TestNetworkXRetrieverProtocol:
    def test_class_exposes_retrieve(self) -> None:
        assert hasattr(NetworkXRetriever, "retrieve")
        assert callable(NetworkXRetriever.retrieve)

    def test_constructed_instance_satisfies_protocol(self, tmp_path: Path) -> None:
        path = _write_graphml(_build_minimal_kg(), tmp_path)
        retriever = NetworkXRetriever(graphml_path=path, k_hop=2)
        assert isinstance(retriever, Retriever)


# -- constructor validation ---------------------------------------------


class TestNetworkXRetrieverConstructorValidation:
    def test_missing_graphml_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="graphml"):
            NetworkXRetriever(graphml_path=tmp_path / "nonexistent.graphml", k_hop=2)

    def test_invalid_k_hop_raises(self, tmp_path: Path) -> None:
        path = _write_graphml(_build_minimal_kg(), tmp_path)
        with pytest.raises(ValueError, match="k_hop"):
            NetworkXRetriever(graphml_path=path, k_hop=0)


# -- retrieve() behaviour ------------------------------------------------


class TestNetworkXRetrieverRetrieve:
    def test_returns_passages_in_score_order(self, tmp_path: Path) -> None:
        path = _write_graphml(_build_minimal_kg(), tmp_path)
        retriever = NetworkXRetriever(graphml_path=path, k_hop=2)
        # "rio" + "enchente" both point at p_a; "barragem" points at p_b.
        # Question mentions only rio/enchente → p_a should rank ahead of p_b.
        results = retriever.retrieve("Quando foi a enchente do rio?", top_k=3)
        assert len(results) >= 1
        assert results[0].chunk_id == "A enchente do rio Uruguai em 2024."
        # Ranks are zero-indexed + consecutive.
        assert [r.rank for r in results] == list(range(len(results)))
        # Scores monotonically non-increasing.
        scores = [r.score for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_top_k_caps_result_size(self, tmp_path: Path) -> None:
        path = _write_graphml(_build_minimal_kg(), tmp_path)
        retriever = NetworkXRetriever(graphml_path=path, k_hop=2)
        results = retriever.retrieve("rio enchente barragem", top_k=1)
        assert len(results) == 1

    def test_retriever_meta_records_score_method(self, tmp_path: Path) -> None:
        path = _write_graphml(_build_minimal_kg(), tmp_path)
        retriever = NetworkXRetriever(graphml_path=path, k_hop=2)
        results = retriever.retrieve("enchente", top_k=1)
        assert results[0].retriever_meta == {"score_method": "node_freq_khop", "k_hop": 2}

    def test_empty_entity_link_returns_empty(self, tmp_path: Path) -> None:
        # Joel's graph-floor guarantee: a question with NO recognizable
        # entities (none of its tokens overlap any node label) must yield []
        # cleanly — no exception, no hallucinated passages.
        path = _write_graphml(_build_minimal_kg(), tmp_path)
        retriever = NetworkXRetriever(graphml_path=path, k_hop=2)
        results = retriever.retrieve("totally unrelated foobar xyzzy", top_k=5)
        assert results == []

    def test_token_overlap_fallback_when_exact_misses(self, tmp_path: Path) -> None:
        # Exact-match seeking a label called "rio" won't find a node literally
        # named "rio" (the node label is "rio Uruguai"). The retriever should
        # tokenize node labels and match per-token, so a single-token query
        # still hits.
        path = _write_graphml(_build_minimal_kg(), tmp_path)
        retriever = NetworkXRetriever(graphml_path=path, k_hop=2)
        results = retriever.retrieve("Uruguai", top_k=3)
        assert any(r.chunk_id.startswith("A enchente do rio Uruguai") for r in results)

    def test_too_common_tokens_dropped_via_postings_cap(self, tmp_path: Path) -> None:
        # If a question token name-matches more than `max_postings` linkable
        # nodes, drop it (IDF-style threshold). Without this, topical words
        # in narrow-domain KGs (e.g. "enchente" in a flood corpus) link to
        # thousands of entities and blow up the ego graph.
        kg = nx.DiGraph()
        # 5 entities all containing the word "common" → 5 postings for "common".
        for i in range(5):
            kg.add_node(f"e_{i}", type="entity", id=f"common label {i}", file_id="p_x")
        # Plus a rare token only on one entity.
        kg.add_node("e_rare", type="entity", id="raretoken specifictoken", file_id="p_x")
        kg.add_node("p_x", type="passage", id="Target passage.", file_id="p_x")
        for n in [f"e_{i}" for i in range(5)] + ["e_rare"]:
            kg.add_edge(n, "p_x")
        path = _write_graphml(kg, tmp_path)

        # max_postings=4 < 5 → "common" gets dropped; "raretoken" survives.
        retriever = NetworkXRetriever(graphml_path=path, k_hop=1, max_postings=4)
        results = retriever.retrieve("common common raretoken", top_k=5)
        # "raretoken" still links e_rare → passage scored via the surviving seed.
        assert any(r.chunk_id == "Target passage." for r in results)

        # With ONLY common tokens (all over the cap), the link is empty.
        empty = retriever.retrieve("common only words", top_k=5)
        assert empty == []

    def test_stopword_only_query_returns_empty(self, tmp_path: Path) -> None:
        # A query composed only of PT/EN stopwords / very short tokens must
        # degenerate to an empty entity link rather than linking to thousands
        # of nodes via common particles. Without this filter, the first
        # real-corpus smoke (test-kg-04, 14k nodes) returned identical
        # top-5 results across very different questions in ~25 min/query
        # because every question's "a" / "que" / "de" linked to most
        # entities.
        path = _write_graphml(_build_minimal_kg(), tmp_path)
        retriever = NetworkXRetriever(graphml_path=path, k_hop=2)
        results = retriever.retrieve("que e a do em", top_k=5)
        assert results == []


# -- k_hop containment ---------------------------------------------------


class TestNetworkXKHopContainment:
    """k_hop bounds how far the subgraph walk reaches."""

    def test_distant_entity_excluded_at_k_hop_1(self, tmp_path: Path) -> None:
        # Build a chain: q_token → mid → far → p_far. With k_hop=1 only
        # `mid` is in the subgraph (so p_far is NOT reached); with k_hop=2
        # `far` joins, but p_far still isn't reached (3 hops away); with
        # k_hop=3 p_far is reached.
        kg = nx.DiGraph()
        kg.add_node("p_far", type="passage", id="Far passage.", file_id="p_far")
        kg.add_node("mid", type="entity", id="mid_token", file_id="p_far")
        kg.add_node("far", type="entity", id="far_token", file_id="p_far")
        kg.add_edge("mid", "far")
        kg.add_edge("far", "p_far")
        path = _write_graphml(kg, tmp_path)

        retriever = NetworkXRetriever(graphml_path=path, k_hop=1)
        assert retriever.retrieve("mid_token", top_k=5) == []

        retriever3 = NetworkXRetriever(graphml_path=path, k_hop=3)
        results3 = retriever3.retrieve("mid_token", top_k=5)
        assert any(r.chunk_id == "Far passage." for r in results3)
