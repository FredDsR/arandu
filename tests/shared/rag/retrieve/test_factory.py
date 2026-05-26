"""Tests for ``shared/rag/retrieve/factory.py`` — per-arm retriever construction.

Locks the dispatch + failure-mode surface. End-to-end retrieve behaviour
(query → passages) is covered by the per-retriever tests in
``tests/shared/rag/retrievers/``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import networkx as nx
import pytest

from arandu.shared.embeddings import EmbedderSettings
from arandu.shared.rag.retrieve.factory import build_retriever
from arandu.shared.rag.retrieve.settings import (
    AtlasRagRetrieveSettings,
    Bm25RetrieveSettings,
    KHopRetrieveSettings,
)
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


class TestAtlasRagArmFailureModes:
    """The atlas_rag arm needs precompute + LLM API key; failures surface clearly."""

    def test_missing_kg_raises_file_not_found(self, tmp_path: Path) -> None:
        # No KG built for this pipeline_id at all.
        with pytest.raises(FileNotFoundError, match="atlas-rag KG outputs not found"):
            build_retriever("atlas_rag", pipeline_id="never_built", base_dir=tmp_path)

    def test_missing_precompute_raises_file_not_found(self, tmp_path: Path) -> None:
        # KG built but `arandu kg-build-retriever-index` was never run.
        _seed_khop_kg(tmp_path, "run_x")  # creates kg/outputs/atlas_output/ + kg_graphml
        with pytest.raises(FileNotFoundError, match="precompute manifest not found"):
            build_retriever("atlas_rag", pipeline_id="run_x", base_dir=tmp_path)

    def test_missing_api_key_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # KG + precompute present; LLM provider is the cloud default but
        # the API key env var is unset. Surface a clear error instead of
        # constructing an unusable LLMClient.
        _seed_khop_kg(tmp_path, "run_x")
        precompute_dir = tmp_path / "run_x" / "kg" / "outputs" / "atlas_output" / "precompute"
        precompute_dir.mkdir(parents=True)
        (precompute_dir / "manifest.json").write_text("{}")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            build_retriever("atlas_rag", pipeline_id="run_x", base_dir=tmp_path)


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


class TestAtlasRagArmConstructs:
    """All prerequisites present → :class:`AtlasRagRetriever` is constructed.

    The real AtlasRagRetriever constructor reads pickles + builds an
    upstream HippoRAGRetriever, which is too heavy for unit tests. We
    patch the class so the test verifies wiring (LLM client + embedder
    plumbing, kwarg names) without touching atlas-rag internals.
    """

    def test_wires_llm_and_embedder_correctly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_khop_kg(tmp_path, "run_x")
        precompute_dir = tmp_path / "run_x" / "kg" / "outputs" / "atlas_output" / "precompute"
        precompute_dir.mkdir(parents=True)
        (precompute_dir / "manifest.json").write_text("{}")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        sentinel_encoder = object()
        with (
            patch(
                "arandu.shared.rag.retrieve.factory.build_embedder",
                return_value=sentinel_encoder,
            ) as mock_build_embedder,
            patch("arandu.shared.rag.retrieve.factory.AtlasRagRetriever") as mock_retriever_cls,
        ):
            build_retriever(
                "atlas_rag",
                pipeline_id="run_x",
                settings=AtlasRagRetrieveSettings(
                    provider="openai",
                    model_id="gemini-2.5-flash",
                    base_url="https://gemini.example/v1/",
                ),
                embedder_settings=EmbedderSettings(model="gemini-embedding-001"),
                base_dir=tmp_path,
            )

        mock_build_embedder.assert_called_once()
        mock_retriever_cls.assert_called_once()
        kwargs = mock_retriever_cls.call_args.kwargs
        assert kwargs["kg_outputs_dir"] == tmp_path / "run_x" / "kg" / "outputs" / "atlas_output"
        assert kwargs["sentence_encoder"] is sentinel_encoder
        assert kwargs["sentence_encoder_model"] == "gemini-embedding-001"
        assert kwargs["llm_model_id"] == "gemini-2.5-flash"
        # llm_client is an openai.OpenAI instance (the .client on LLMClient).
        # We don't assert its full type — just that it was passed.
        assert kwargs["llm_client"] is not None


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
