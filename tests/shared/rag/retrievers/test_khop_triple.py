"""Tests for ``shared/rag/retrievers/khop_triple.py`` — triple-injection variant."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import pytest

from arandu.shared.rag.protocol import Retriever
from arandu.shared.rag.retrievers.khop_triple import KHopTripleRetriever

if TYPE_CHECKING:
    from pathlib import Path


# -- fixtures ------------------------------------------------------------


def _build_triple_kg() -> nx.DiGraph:
    """Synthetic atlas-rag-shaped KG covering all endpoint-type cases.

    Tests assert two exclusions from the triple emission:
    - **Passage-incident edges** (the passage's ``id`` is a 2 KB
      transcription blob; including it would defeat structured-context).
    - **Concept-incident edges** (`has_concept` schema-induction
      artifacts where the concept node is a hub; including these floods
      the top-k with type-attachment triples regardless of question — see
      the smoke against `test-kg-04` on 2026-05-23).

    Graph edges + which endpoint-type rules they exercise:

    - ``e_maria --[vive_em]--> e_barra`` (entity → entity, INCLUDED)
    - ``e_maria --[trabalha_com]--> e_pesca`` (entity → entity, INCLUDED)
    - ``e_barra --[localizado_em]--> e_rio`` (entity → entity, INCLUDED)
    - ``ev_enchente --[afetou]--> e_barra`` (event → entity, INCLUDED)
    - ``e_maria --[mentioned_in]--> p_a`` (entity → passage, EXCLUDED)
    - ``e_pesca --[has_concept]--> c_artesanal`` (entity → concept, EXCLUDED)
    """
    kg = nx.DiGraph()
    # Entities, events (linkable + valid triple endpoints)
    kg.add_node("e_maria", type="entity", id="Maria da Silva")
    kg.add_node("e_barra", type="entity", id="Barra do Ribeiro")
    kg.add_node("e_pesca", type="entity", id="pesca artesanal")
    kg.add_node("e_rio", type="entity", id="rio Uruguai")
    kg.add_node("ev_enchente", type="event", id="enchente de 2024")
    # Concept (linkable for the entity-link stage, but EXCLUDED from triple endpoints)
    kg.add_node("c_artesanal", type="concept", id="atividade artesanal")
    # Passage (also excluded from triple endpoints)
    kg.add_node("p_a", type="passage", id="[Contexto]\nLong passage text here.")

    kg.add_edge("e_maria", "e_barra", relation="vive_em")
    kg.add_edge("e_maria", "e_pesca", relation="trabalha_com")
    kg.add_edge("e_barra", "e_rio", relation="localizado_em")
    kg.add_edge("e_maria", "p_a", relation="mentioned_in")  # passage edge — EXCLUDED
    kg.add_edge("ev_enchente", "e_barra", relation="afetou")
    kg.add_edge("e_pesca", "c_artesanal", relation="has_concept")  # concept edge — EXCLUDED
    return kg


def _write_kg_layout(kg: nx.DiGraph, tmp_path: Path) -> Path:
    """Lay out a minimal ``atlas_output/kg_graphml/`` tree.

    Unlike the passage NetworkX retriever, this one needs only the
    graphml — no ``kg_extraction/`` JSONL is read because triples come
    from edge attributes, not from JSONL records.
    """
    kg_outputs_dir = tmp_path / "atlas_output"
    graphml_dir = kg_outputs_dir / "kg_graphml"
    graphml_dir.mkdir(parents=True)
    nx.write_graphml(kg, graphml_dir / "transcriptions.json_graph.graphml")
    return kg_outputs_dir


# -- naming + Protocol ---------------------------------------------------


class TestKHopTripleRetrieverId:
    def test_default_id(self) -> None:
        assert KHopTripleRetriever.RETRIEVER_FAMILY == "khop"
        assert KHopTripleRetriever.DEFAULT_RETRIEVER_ID == "khop_triple"


class TestKHopTripleRetrieverProtocol:
    def test_class_exposes_retrieve(self) -> None:
        assert hasattr(KHopTripleRetriever, "retrieve")
        assert callable(KHopTripleRetriever.retrieve)

    def test_instance_satisfies_protocol(self, tmp_path: Path) -> None:
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=2)
        assert isinstance(retriever, Retriever)


# -- constructor validation ---------------------------------------------


class TestKHopTripleConstructorValidation:
    def test_missing_graphml_raises(self, tmp_path: Path) -> None:
        kg_outputs_dir = tmp_path / "atlas_output"
        (kg_outputs_dir / "kg_graphml").mkdir(parents=True)
        # graphml file itself is missing
        with pytest.raises(FileNotFoundError, match="graphml"):
            KHopTripleRetriever(kg_outputs_dir=kg_outputs_dir, k_hop=2)

    def test_invalid_k_hop_raises(self, tmp_path: Path) -> None:
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        with pytest.raises(ValueError, match="k_hop"):
            KHopTripleRetriever(kg_outputs_dir=path, k_hop=0)

    def test_invalid_top_k_seeds_raises(self, tmp_path: Path) -> None:
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        with pytest.raises(ValueError, match="top_k_seeds"):
            KHopTripleRetriever(kg_outputs_dir=path, top_k_seeds=0)


# -- retrieve() behaviour ------------------------------------------------


class TestKHopTripleRetrieve:
    def test_top_k_zero_returns_empty_not_indexerror(self, tmp_path: Path) -> None:
        # Defensive: if the caller passes top_k <= 0, the retriever must
        # return [] cleanly. Without the guard, `ranked = triples[:0]` is
        # empty and `ranked[0]` would raise IndexError on the next line.
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=2)
        assert retriever.retrieve("Onde Maria mora?", top_k=0) == []

    def test_emits_linearized_triples_via_payload(self, tmp_path: Path) -> None:
        # Question links to "Maria" → seeds at e_maria. k_hop=2 reaches her
        # entity neighbours. Triples emitted as payload, chunk_id is a
        # synthesised sha1-keyed handle.
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("Onde Maria mora?", top_k=10)

        assert results, "expected non-empty triple list for in-KG question"
        for r in results:
            assert r.payload is not None, "every result must carry a triple in payload"
            assert " --[" in r.payload and "]--> " in r.payload, (
                f"payload must be linearized as `[type] head --[rel]--> [type] tail`, "
                f"got {r.payload!r}"
            )
            assert r.chunk_id.startswith("triple:"), (
                f"chunk_id should be `triple:<sha>`, got {r.chunk_id!r}"
            )

    def test_excludes_passage_node_triples(self, tmp_path: Path) -> None:
        # Even though e_maria → p_a exists as an edge, the result must NOT
        # include any triple where a passage node is an endpoint. Passage
        # `id` is the full transcription text with atlas-rag header, which
        # would produce multi-paragraph "triples" that defeat the
        # structured-context purpose.
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("Maria", top_k=20)

        for r in results:
            assert r.payload is not None
            assert "[passage]" not in r.payload, (
                "passage-incident edges must be filtered before triple emission"
            )
            assert "[Contexto" not in r.payload, "passage text must not leak into triple payload"

    def test_excludes_concept_node_triples(self, tmp_path: Path) -> None:
        # `has_concept` edges link entities to atlas-rag's induced schema
        # types (`feição geográfica`, `recurso natural`, ...). Concept
        # nodes are hubs (every entity attaches to them), so endpoint-
        # degree scoring otherwise floods the top-k with type-attachment
        # triples for ANY question — that's exactly what the first smoke
        # against `test-kg-04` produced (identical concept triples for
        # three different questions). The retriever must filter them out
        # to preserve the methodology §6.4 "semantic relation between
        # entities" paradigm.
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=2)
        # Question links to e_pesca via "pesca", which neighbours the
        # concept node c_artesanal via `has_concept`. The concept edge
        # must be skipped.
        results = retriever.retrieve("pesca", top_k=20)

        for r in results:
            assert r.payload is not None
            assert "[concept]" not in r.payload, (
                "concept-incident edges (atlas-rag schema-induction artifacts) "
                "must be filtered before triple emission"
            )
            assert "has_concept" not in r.payload, (
                "the `has_concept` relation should never surface in scored triples"
            )

    def test_rank_and_score_shape(self, tmp_path: Path) -> None:
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("Maria Barra", top_k=5)
        assert results
        assert [r.rank for r in results] == list(range(len(results)))
        scores = [r.score for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        # Top score is normalised to 1.0 (max-relative scaling).
        assert scores[0] == pytest.approx(1.0)

    def test_retriever_meta_records_score_method(self, tmp_path: Path) -> None:
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("Maria", top_k=1)
        assert results[0].retriever_meta == {
            "score_method": "seed_proximity",
            "k_hop": 2,
        }

    def test_top_k_caps_result_size(self, tmp_path: Path) -> None:
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=2)
        # The fixture has 5 entity-only triples; asking for 2 must clamp.
        results = retriever.retrieve("Maria Barra rio", top_k=2)
        assert len(results) == 2

    def test_empty_entity_link_returns_empty(self, tmp_path: Path) -> None:
        # Graph-floor guarantee — same as the passage NetworkX arm.
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=2)
        assert retriever.retrieve("totally unrelated foobar xyzzy", top_k=5) == []

    def test_chunk_id_is_synthetic_not_offset_resolvable(self, tmp_path: Path) -> None:
        # Triples don't have source offsets — their chunk_id is a sha1-keyed
        # synthetic handle, NOT an offset-resolvable reference. Downstream
        # judges that consult passage_offsets.json must skip records where
        # `payload` is set (per the schema docstring).
        path = _write_kg_layout(_build_triple_kg(), tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("Maria", top_k=3)
        assert results
        for r in results:
            assert r.chunk_id.startswith("triple:")
            assert ":" not in r.chunk_id.removeprefix("triple:"), (
                "synthetic chunk_id must not look like a `<file_id>:<idx>` "
                "passage_id (would collide with passage-namespace consumers)"
            )

    def test_self_loops_excluded(self, tmp_path: Path) -> None:
        # atlas-rag occasionally emits self-loops (`A --[rel]--> A`).
        # They're structurally weird and methodologically uninformative —
        # a relation from an entity to itself doesn't carry cross-entity
        # semantic content. The first proximity-scored smoke against
        # `test-kg-04` returned `Dona Gilda --[envolve]--> Dona Gilda` as
        # the top result because self-loops at a seed have proximity 0+0;
        # this exclusion ensures they never surface in the output.
        kg = nx.DiGraph()
        kg.add_node("e_a", type="entity", id="Alpha")
        kg.add_node("e_b", type="entity", id="Beta")
        kg.add_edge("e_a", "e_a", relation="self_referential")  # self-loop
        kg.add_edge("e_a", "e_b", relation="connects_to")  # real edge
        path = _write_kg_layout(kg, tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=1)
        results = retriever.retrieve("Alpha", top_k=5)
        assert results
        # No triple should have Alpha on both sides.
        for r in results:
            assert "Alpha --[self_referential]--> [entity] Alpha" not in r.payload

    def test_seed_proximity_ranks_seed_incident_edges_first(self, tmp_path: Path) -> None:
        # Edges directly touching a seed should outrank edges 1+ hops away.
        # This pins the new scoring contract (vs the prior endpoint-degree
        # scoring, which favoured graph hubs regardless of question relevance).
        kg = nx.DiGraph()
        # Seed entity will be `e_seed` (matched via "seedlabel").
        kg.add_node("e_seed", type="entity", id="seedlabel rare")
        # Direct neighbour — edge touches a seed (proximity 0+1 = 1 → score 0.5).
        kg.add_node("e_near", type="entity", id="near entity")
        kg.add_edge("e_seed", "e_near", relation="touches_seed")
        # One-hop-removed neighbour — edge between two non-seed nodes
        # both 1 hop away (proximity 1+1 = 2 → score 0.33).
        kg.add_node("e_far", type="entity", id="far entity")
        kg.add_edge("e_near", "e_far", relation="far_from_seed")
        # A high-degree hub OUTSIDE the seed's neighbourhood — old scoring
        # would have surfaced this; new scoring should not.
        kg.add_node("e_hub", type="entity", id="hub entity")
        for i in range(5):
            kg.add_node(f"e_filler_{i}", type="entity", id=f"filler{i}")
            kg.add_edge("e_hub", f"e_filler_{i}", relation="hub_fanout")
        # Connect hub to the rest via a tenuous link so the k_hop traversal reaches it.
        kg.add_edge("e_far", "e_hub", relation="distant_link")
        path = _write_kg_layout(kg, tmp_path)

        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=3)
        results = retriever.retrieve("seedlabel", top_k=10)
        assert results

        # The very first result must be the seed-incident edge.
        assert results[0].payload == (
            "[entity] seedlabel rare --[touches_seed]--> [entity] near entity"
        )
        # The hub_fanout edges (graph-hub artefacts) must rank below seed-incident
        # ones. Confirm none of the top-3 are hub_fanout.
        for r in results[:3]:
            assert "hub_fanout" not in r.payload

    def test_unknown_relation_falls_back_to_default(self, tmp_path: Path) -> None:
        # If an edge lacks a `relation` attribute, the retriever uses a
        # `related_to` placeholder rather than failing or emitting a
        # malformed triple.
        kg = nx.DiGraph()
        kg.add_node("e_a", type="entity", id="Alpha")
        kg.add_node("e_b", type="entity", id="Beta")
        kg.add_edge("e_a", "e_b")  # no `relation` attr
        path = _write_kg_layout(kg, tmp_path)
        retriever = KHopTripleRetriever(kg_outputs_dir=path, k_hop=1)
        results = retriever.retrieve("Alpha", top_k=5)
        assert results
        assert any("related_to" in r.payload for r in results)


def test_top_k_seeds_param_and_no_max_postings() -> None:
    import inspect

    from arandu.shared.rag.retrievers.khop_triple import KHopTripleRetriever

    sig = inspect.signature(KHopTripleRetriever.__init__)
    assert "top_k_seeds" in sig.parameters
    assert "max_postings" not in sig.parameters
