from __future__ import annotations

import networkx as nx
import pytest

from arandu.shared.rag.retrievers import _khop_common as kc


def _kg(labels: dict[str, str]) -> nx.DiGraph:
    """Build a tiny KG; labels maps node_id -> label, all type 'entity'."""
    g = nx.DiGraph()
    for node_id, label in labels.items():
        g.add_node(node_id, id=label, type="entity")
    return g


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


class TestLinkEntities:
    def test_rare_token_seed_outranks_common_token_seed(self) -> None:
        g = _kg(
            {
                "n1": "comum coisa",
                "n2": "comum outra",
                "n3": "comum mais",
                "n4": "valverde",
            }
        )
        idx, n = kc.build_label_index(g)
        seeds = kc.link_entities("valverde comum", idx, n, top_k_seeds=2)
        assert "n4" in seeds

    def test_top_k_budget_caps_seed_count(self) -> None:
        g = _kg({f"n{i}": "comum" for i in range(10)})
        idx, n = kc.build_label_index(g)
        seeds = kc.link_entities("comum", idx, n, top_k_seeds=3)
        assert len(seeds) == 3

    def test_common_only_query_still_seeds(self) -> None:
        g = _kg({f"n{i}": "pesca" for i in range(500)})
        idx, n = kc.build_label_index(g)
        seeds = kc.link_entities("pesca", idx, n, top_k_seeds=50)
        assert len(seeds) == 50

    def test_no_match_returns_empty(self) -> None:
        g = _kg({"n1": "barra", "n2": "uruguai"})
        idx, n = kc.build_label_index(g)
        assert kc.link_entities("inexistente", idx, n, top_k_seeds=50) == []


class TestSubgraphNodeDistances:
    def test_seed_is_distance_zero_neighbor_one(self) -> None:
        g = nx.DiGraph()
        g.add_node("s", id="seed", type="entity")
        g.add_node("a", id="a", type="entity")
        g.add_edge("s", "a", relation="r")
        dist = kc.subgraph_node_distances(g, ["s"], k_hop=2)
        assert dist["s"] == 0
        assert dist["a"] == 1

    def test_beyond_k_hop_excluded(self) -> None:
        g = nx.DiGraph()
        for n in ("s", "a", "b", "c"):
            g.add_node(n, id=n, type="entity")
        g.add_edge("s", "a", relation="r")
        g.add_edge("a", "b", relation="r")
        g.add_edge("b", "c", relation="r")
        dist = kc.subgraph_node_distances(g, ["s"], k_hop=2)
        assert "b" in dist and dist["b"] == 2
        assert "c" not in dist

    def test_min_distance_over_multiple_seeds(self) -> None:
        g = nx.DiGraph()
        for n in ("s1", "s2", "a", "x"):
            g.add_node(n, id=n, type="entity")
        g.add_edge("s1", "a", relation="r")
        g.add_edge("a", "s2", relation="r")
        g.add_edge("s1", "x", relation="r")
        dist = kc.subgraph_node_distances(g, ["s1", "s2"], k_hop=2)
        assert dist["a"] == 1
        assert dist["s1"] == 0 and dist["s2"] == 0

    def test_empty_seeds_returns_empty(self) -> None:
        g = nx.DiGraph()
        g.add_node("s", id="seed", type="entity")
        assert kc.subgraph_node_distances(g, [], k_hop=2) == {}
