"""Tests for ``shared/rag/retrieve/factory.py`` — per-arm retriever construction.

Locks the dispatch + failure-mode surface. End-to-end retrieve behaviour
(query → passages) is covered by the per-retriever tests in
``tests/shared/rag/retrievers/``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import networkx as nx
import pytest

from arandu.shared.rag.retrieve.factory import build_retriever
from arandu.shared.rag.retrieve.settings import Bm25RetrieveSettings, KHopRetrieveSettings
from arandu.shared.rag.retrievers.khop_subgraph import KHopSubgraphRetriever
from arandu.shared.rag.retrievers.khop_triple import KHopTripleRetriever
from arandu.shared.rag.retrievers.null import NullRetriever

if TYPE_CHECKING:
    from pathlib import Path


class TestNullArm:
    def test_returns_null_retriever_without_paths(self, tmp_path: Path) -> None:
        # Null arm needs no on-disk artifacts — useful smoke for the
        # CLI's "always at least the null arm runs" contract.
        retriever = build_retriever("null", pipeline_id="any", base_dir=tmp_path)
        assert isinstance(retriever, NullRetriever)


class TestAtlasRagArmRejected:
    def test_atlas_rag_raises_with_followup_pr_hint(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not wired in this PR"):
            build_retriever("atlas_rag", pipeline_id="any", base_dir=tmp_path)


class TestUnknownArm:
    def test_unknown_arm_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown arm"):
            build_retriever("does_not_exist", pipeline_id="any", base_dir=tmp_path)  # type: ignore[arg-type]


def _seed_khop_kg(base: Path, pipeline_id: str, keyword: str = "transcriptions.json") -> Path:
    """Lay out a minimal atlas-rag KG tree with a real graphml file."""
    kg_outputs_dir = base / pipeline_id / "kg" / "outputs" / "atlas_output"
    graphml_dir = kg_outputs_dir / "kg_graphml"
    extraction_dir = kg_outputs_dir / "kg_extraction"
    graphml_dir.mkdir(parents=True)
    extraction_dir.mkdir(parents=True)

    kg = nx.DiGraph()
    kg.add_node("e_a", type="entity", id="entidade a", file_id="src_a")
    kg.add_node("p_a", type="passage", id="texto teste", file_id="src_a")
    kg.add_edge("e_a", "p_a", relation="mentions")
    nx.write_graphml(kg, graphml_dir / f"{keyword}_graph.graphml")

    (extraction_dir / "qwen3.json").write_text(
        json.dumps({"id": "src_a", "original_text": "texto teste"}) + "\n"
    )
    return kg_outputs_dir


class TestKHopArms:
    def test_khop_passage_constructs_retriever(self, tmp_path: Path) -> None:
        _seed_khop_kg(tmp_path, "run_x")
        retriever = build_retriever(
            "khop_passage",
            pipeline_id="run_x",
            settings=KHopRetrieveSettings(k_hop=2, max_postings=10),
            base_dir=tmp_path,
        )
        assert isinstance(retriever, KHopSubgraphRetriever)

    def test_khop_triple_constructs_retriever(self, tmp_path: Path) -> None:
        _seed_khop_kg(tmp_path, "run_x")
        retriever = build_retriever(
            "khop_triple",
            pipeline_id="run_x",
            settings=KHopRetrieveSettings(k_hop=2, max_postings=10),
            base_dir=tmp_path,
        )
        assert isinstance(retriever, KHopTripleRetriever)

    def test_khop_missing_kg_raises(self, tmp_path: Path) -> None:
        # No KG built for this pipeline_id at all.
        with pytest.raises(FileNotFoundError, match="atlas-rag KG outputs not found"):
            build_retriever(
                "khop_passage",
                pipeline_id="nonexistent",
                base_dir=tmp_path,
            )


class TestBm25Arm:
    def test_missing_chunks_raises(self, tmp_path: Path) -> None:
        # No chunks for this chunker_id.
        with pytest.raises(FileNotFoundError, match="Chunks not found"):
            build_retriever(
                "bm25",
                pipeline_id="run_x",
                settings=Bm25RetrieveSettings(chunker_id="cep_4k"),
                base_dir=tmp_path,
            )
