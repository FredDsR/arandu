"""Tests for ``shared/rag/retrievers/khop_subgraph.py`` — graph sanity baseline (spec §4.5).

Joel's "graph-floor" arm: entity-link → k-hop ego graph → passage-mention
frequency scoring. Tests cover constructor validation, retrieve() behaviour,
chunk_id bridging, stopword/postings-cap filtering, and k_hop containment.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import networkx as nx
import pytest

from arandu.shared.rag.protocol import Retriever
from arandu.shared.rag.retrievers.khop_subgraph import KHopSubgraphRetriever

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


def _write_kg_layout(
    kg: nx.DiGraph,
    tmp_path: Path,
    *,
    keyword: str = "transcriptions.json",
    kg_extraction_records: list[dict] | None = None,
) -> Path:
    """Lay out a fake atlas-rag-shaped ``kg/outputs/atlas_output/`` tree.

    Writes ``kg_graphml/<keyword>_graph.graphml`` + populates
    ``kg_extraction/`` from ``kg_extraction_records`` (one JSONL file). If
    ``kg_extraction_records`` is None, derives records from the KG's
    passage nodes — one record per passage with ``id`` derived from the
    ``file_id`` attribute (so the synthesized passage_id is
    ``<file_id>:0`` for each).

    Returns the ``kg_outputs_dir`` (the parent of ``kg_graphml/`` and
    ``kg_extraction/``).
    """
    kg_outputs_dir = tmp_path / "atlas_output"
    graphml_dir = kg_outputs_dir / "kg_graphml"
    extraction_dir = kg_outputs_dir / "kg_extraction"
    graphml_dir.mkdir(parents=True)
    extraction_dir.mkdir(parents=True)

    graphml_path = graphml_dir / f"{keyword}_graph.graphml"
    nx.write_graphml(kg, graphml_path)

    if kg_extraction_records is None:
        kg_extraction_records = [
            {"id": attrs["file_id"], "original_text": attrs["id"]}
            for _, attrs in kg.nodes(data=True)
            if attrs.get("type") == "passage"
        ]
    (extraction_dir / "qwen3_extraction.json").write_text(
        "\n".join(json.dumps(r) for r in kg_extraction_records) + "\n"
    )
    return kg_outputs_dir


# -- naming + Protocol --------------------------------------------------


class TestKHopSubgraphRetrieverId:
    """`retriever_id` follows the family-prefix convention from BM25 + atlas-rag."""

    def test_default_id(self) -> None:
        assert KHopSubgraphRetriever.RETRIEVER_FAMILY == "khop"
        assert KHopSubgraphRetriever.DEFAULT_RETRIEVER_ID == "khop_passage"


class TestKHopSubgraphRetrieverProtocol:
    def test_class_exposes_retrieve(self) -> None:
        assert hasattr(KHopSubgraphRetriever, "retrieve")
        assert callable(KHopSubgraphRetriever.retrieve)

    def test_constructed_instance_satisfies_protocol(self, tmp_path: Path) -> None:
        path = _write_kg_layout(_build_minimal_kg(), tmp_path)
        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=2)
        assert isinstance(retriever, Retriever)


# -- constructor validation ---------------------------------------------


class TestKHopSubgraphRetrieverConstructorValidation:
    def test_missing_graphml_raises(self, tmp_path: Path) -> None:
        # Build a kg_outputs_dir with the kg_extraction subdir present but
        # NO graphml. The constructor must surface the missing graphml
        # cleanly (rather than crashing later when nx.read_graphml is called).
        kg_outputs_dir = tmp_path / "atlas_output"
        (kg_outputs_dir / "kg_extraction").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="graphml"):
            KHopSubgraphRetriever(kg_outputs_dir=kg_outputs_dir, k_hop=2)

    def test_missing_kg_extraction_dir_raises(self, tmp_path: Path) -> None:
        # If only the graphml exists, no passage_id bridge can be built →
        # retrieval would have to fall back to opaque KG node hashes,
        # defeating the spec's "stable chunk_id" contract. Fail fast.
        kg_outputs_dir = tmp_path / "atlas_output"
        graphml_dir = kg_outputs_dir / "kg_graphml"
        graphml_dir.mkdir(parents=True)
        (graphml_dir / "transcriptions.json_graph.graphml").write_bytes(
            b'<?xml version="1.0"?><graphml/>'
        )
        with pytest.raises(FileNotFoundError, match="kg_extraction"):
            KHopSubgraphRetriever(kg_outputs_dir=kg_outputs_dir, k_hop=2)

    def test_invalid_k_hop_raises(self, tmp_path: Path) -> None:
        path = _write_kg_layout(_build_minimal_kg(), tmp_path)
        with pytest.raises(ValueError, match="k_hop"):
            KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=0)


# -- retrieve() behaviour ------------------------------------------------


class TestKHopSubgraphRetrieverRetrieve:
    def test_returns_passages_in_score_order(self, tmp_path: Path) -> None:
        path = _write_kg_layout(_build_minimal_kg(), tmp_path)
        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=2)
        # "rio" + "enchente" both point at p_a; "barragem" points at p_b.
        # Question mentions only rio/enchente → p_a should rank ahead of p_b.
        results = retriever.retrieve("Quando foi a enchente do rio?", top_k=3)
        assert len(results) >= 1
        # chunk_id is the atlas-rag synthesized passage_id (`<file_id>:<idx>`),
        # NOT the passage text — same namespace as the passage_offsets sidecar
        # so downstream judges can join.
        assert results[0].chunk_id == "p_a:0"
        # Ranks are zero-indexed + consecutive.
        assert [r.rank for r in results] == list(range(len(results)))
        # Scores monotonically non-increasing.
        scores = [r.score for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_top_k_caps_result_size(self, tmp_path: Path) -> None:
        path = _write_kg_layout(_build_minimal_kg(), tmp_path)
        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("rio enchente barragem", top_k=1)
        assert len(results) == 1

    def test_top_k_zero_returns_empty(self, tmp_path: Path) -> None:
        # Uniform contract across all retriever arms: top_k <= 0 → [].
        # Previously, the build-then-cap loop appended a record on iteration
        # 1, then `len(ranked) >= 0` broke — but the record was already in
        # the list, so retrieve() silently returned 1 record when 0 were
        # requested. Now guarded with an early `if top_k <= 0: return []`.
        path = _write_kg_layout(_build_minimal_kg(), tmp_path)
        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=2)
        assert retriever.retrieve("rio enchente", top_k=0) == []

    def test_chunk_id_is_atlas_passage_id_not_text(self, tmp_path: Path) -> None:
        # Per the `RetrievedPassage.chunk_id` docstring it must be a
        # "reference into the source ChunkSet" — i.e. a stable identifier,
        # not the passage text. The retriever bridges KG passage nodes to
        # atlas-rag's synthesized `<source_file_id>:<chunk_index>` via the
        # kg_extraction JSONL. Verify chunk_id matches that namespace AND
        # is NOT the passage text.
        path = _write_kg_layout(_build_minimal_kg(), tmp_path)
        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("rio Uruguai enchente", top_k=5)
        assert results, "expected non-empty results for in-KG question"
        for r in results:
            assert ":" in r.chunk_id, (
                f"chunk_id should be `<source_file_id>:<chunk_index>`, got {r.chunk_id!r}"
            )
            # No passage text should leak through.
            assert "[Contexto" not in r.chunk_id
            assert "A enchente" not in r.chunk_id

    def test_payload_carries_source_prose(self, tmp_path: Path) -> None:
        # khop_passage carries the source passage inline in `payload` so the
        # Answerer doesn't re-resolve `chunk_id` through passage_offsets.json at
        # answer time. `chunk_id` stays the joinable id; `payload_is_prose=True`
        # keeps source_recovery (prose token-containment) applicable, unlike the
        # triple arm's non-prose payload.
        path = _write_kg_layout(_build_minimal_kg(), tmp_path)
        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("rio Uruguai enchente", top_k=5)
        assert results, "expected non-empty results for in-KG question"
        for r in results:
            assert r.payload, "expected source prose carried in payload"
            assert r.payload != r.chunk_id, "payload must be the text, not the id"
            assert r.payload_is_prose is True

    def test_passage_with_no_kg_extraction_record_dropped(self, tmp_path: Path) -> None:
        # If a KG passage node has no matching kg_extraction record (corpus
        # drift between KG build and JSONL on disk), the retriever drops it
        # rather than emitting a `RetrievedPassage` with an opaque hash that
        # can't be joined with the offset sidecar downstream judges consult.
        kg = nx.DiGraph()
        kg.add_node("p_in_jsonl", type="passage", id="text in jsonl", file_id="p_in_jsonl")
        kg.add_node("p_orphan", type="passage", id="text NOT in jsonl", file_id="p_orphan")
        kg.add_node("e_a", type="entity", id="alpha entity", file_id="p_in_jsonl")
        kg.add_node("e_b", type="entity", id="beta entity", file_id="p_orphan")
        kg.add_edge("e_a", "p_in_jsonl")
        kg.add_edge("e_b", "p_orphan")
        # JSONL only includes p_in_jsonl — p_orphan has no bridge.
        path = _write_kg_layout(
            kg,
            tmp_path,
            kg_extraction_records=[{"id": "p_in_jsonl", "original_text": "text in jsonl"}],
        )

        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("alpha beta", top_k=5)
        # Only p_in_jsonl makes it through.
        assert [r.chunk_id for r in results] == ["p_in_jsonl:0"]

    def test_retriever_meta_records_score_method(self, tmp_path: Path) -> None:
        path = _write_kg_layout(_build_minimal_kg(), tmp_path)
        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=2, max_postings=50)
        results = retriever.retrieve("enchente", top_k=1)
        assert results[0].retriever_meta == {
            "score_method": "node_freq_khop",
            "k_hop": 2,
            "max_postings": 50,
        }

    def test_empty_entity_link_returns_empty(self, tmp_path: Path) -> None:
        # Joel's graph-floor guarantee: a question with NO recognizable
        # entities (none of its tokens overlap any node label) must yield []
        # cleanly — no exception, no hallucinated passages.
        path = _write_kg_layout(_build_minimal_kg(), tmp_path)
        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("totally unrelated foobar xyzzy", top_k=5)
        assert results == []

    def test_token_overlap_fallback_when_exact_misses(self, tmp_path: Path) -> None:
        # Exact-match seeking a label called "rio" won't find a node literally
        # named "rio" (the node label is "rio Uruguai"). The retriever should
        # tokenize node labels and match per-token, so a single-token query
        # still hits.
        path = _write_kg_layout(_build_minimal_kg(), tmp_path)
        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("Uruguai", top_k=3)
        assert any(r.chunk_id == "p_a:0" for r in results)

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
        path = _write_kg_layout(kg, tmp_path)

        # max_postings=4 < 5 → "common" gets dropped; "raretoken" survives.
        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=1, max_postings=4)
        results = retriever.retrieve("common common raretoken", top_k=5)
        # "raretoken" still links e_rare → passage scored via the surviving seed.
        assert any(r.chunk_id == "p_x:0" for r in results)

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
        path = _write_kg_layout(_build_minimal_kg(), tmp_path)
        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=2)
        results = retriever.retrieve("que e a do em", top_k=5)
        assert results == []


# -- k_hop containment ---------------------------------------------------


class TestKHopContainment:
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
        path = _write_kg_layout(kg, tmp_path)

        retriever = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=1)
        assert retriever.retrieve("mid_token", top_k=5) == []

        retriever3 = KHopSubgraphRetriever(kg_outputs_dir=path, k_hop=3)
        results3 = retriever3.retrieve("mid_token", top_k=5)
        assert any(r.chunk_id == "p_far:0" for r in results3)
