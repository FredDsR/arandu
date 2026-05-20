"""Tests for ``shared/rag/retrievers/atlas_rag.py`` — atlas-rag HippoRAG wrapper (spec §4.4).

The wrapper logic is what's tested here: adapter shape, validation, and
behaviour at the boundary with atlas-rag. Atlas-rag's own retrieval
quality is NOT re-tested — that's upstream's job. Real-KG smoke is
exercised separately via a script against ``results/test-kg-04``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from arandu.shared.rag.protocol import Retriever
from arandu.shared.rag.retrievers.atlas_rag import (
    PRECOMPUTE_DIR_NAME,
    AtlasRagRetriever,
    _atlas_results_to_retrieved_passages,
)

if TYPE_CHECKING:
    from pathlib import Path


# -- pure adapter ---------------------------------------------------------


class TestAtlasResultsToRetrievedPassages:
    """The adapter from atlas-rag's `(passage_id, score)` pairs to our schema."""

    def test_emits_one_passage_per_id(self) -> None:
        passages = _atlas_results_to_retrieved_passages(
            [("src_a:0", 0.91), ("src_b:1", 0.72), ("src_c:0", 0.05)]
        )
        assert len(passages) == 3
        assert [p.chunk_id for p in passages] == ["src_a:0", "src_b:1", "src_c:0"]

    def test_ranks_are_zero_indexed_consecutive(self) -> None:
        passages = _atlas_results_to_retrieved_passages([("x", 0.5), ("y", 0.3), ("z", 0.1)])
        assert [p.rank for p in passages] == [0, 1, 2]

    def test_scores_pass_through_as_float(self) -> None:
        passages = _atlas_results_to_retrieved_passages([("x", 0.91), ("y", 0.42)])
        assert passages[0].score == pytest.approx(0.91)
        assert passages[1].score == pytest.approx(0.42)
        assert all(isinstance(p.score, float) for p in passages)

    def test_retriever_meta_records_score_method(self) -> None:
        passages = _atlas_results_to_retrieved_passages([("x", 0.9)])
        assert passages[0].retriever_meta == {"score_method": "hipporag"}

    def test_empty_input_returns_empty_list(self) -> None:
        assert _atlas_results_to_retrieved_passages([]) == []


# -- constructor validation ----------------------------------------------


class TestAtlasRagRetrieverConstructorValidation:
    """Surfaces failure modes before the heavy atlas-rag machinery runs."""

    def test_missing_kg_outputs_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="kg outputs"):
            AtlasRagRetriever(
                kg_outputs_dir=tmp_path / "nonexistent",
                llm_client=MagicMock(),
                sentence_encoder=MagicMock(),
                sentence_encoder_model="m",
            )

    def test_missing_precompute_dir_raises(self, tmp_path: Path) -> None:
        kg_outputs_dir = tmp_path / "atlas_output"
        kg_outputs_dir.mkdir()
        # Has the dir but no precompute/ subdir — index has not been built.
        with pytest.raises(FileNotFoundError, match="precompute"):
            AtlasRagRetriever(
                kg_outputs_dir=kg_outputs_dir,
                llm_client=MagicMock(),
                sentence_encoder=MagicMock(),
                sentence_encoder_model="m",
            )

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        kg_outputs_dir = tmp_path / "atlas_output"
        precompute = kg_outputs_dir / PRECOMPUTE_DIR_NAME
        precompute.mkdir(parents=True)
        # Has the dir but no manifest — atlas-rag's create_embeddings_and_index
        # was never run via our wrapper.
        with pytest.raises(FileNotFoundError, match="manifest"):
            AtlasRagRetriever(
                kg_outputs_dir=kg_outputs_dir,
                llm_client=MagicMock(),
                sentence_encoder=MagicMock(),
                sentence_encoder_model="m",
            )


# -- retriever_id naming -------------------------------------------------


class TestAtlasRagRetrieverId:
    """`retriever_id` follows BM25's family-prefix convention."""

    def test_default_id_uses_family_prefix(self) -> None:
        # `atlas_rag_hipporag` distinguishes from a future atlas_rag arm
        # (e.g. SimpleTextRetriever) that might share the chunker view.
        assert AtlasRagRetriever.RETRIEVER_FAMILY == "atlas_rag"
        assert AtlasRagRetriever.DEFAULT_RETRIEVER_ID == "atlas_rag_hipporag"


# -- Retriever Protocol conformance --------------------------------------


class TestAtlasRagRetrieverProtocol:
    """The wrapper satisfies the `Retriever` Protocol structural contract."""

    def test_class_advertises_protocol_attrs(self) -> None:
        # Structural check: class-level retriever_id attribute + retrieve method.
        # (Full Protocol isinstance() needs a constructed instance; covered by
        # the smoke script against test-kg-04.)
        assert hasattr(AtlasRagRetriever, "retrieve")
        assert callable(AtlasRagRetriever.retrieve)


# -- retrieve() adapter behaviour (with mocked inner retriever) ----------


class TestAtlasRagRetrieverRetrieve:
    """`retrieve()` calls the score-capturing path and adapts the result."""

    def test_returns_retrieved_passages_in_descending_score_order(self) -> None:
        # Bypass __init__ — we are testing only the adapter glue, not the
        # heavy PPR pipeline. `_retrieve_with_scores` is the seam between
        # the upstream HippoRAG call and our schema.
        instance = AtlasRagRetriever.__new__(AtlasRagRetriever)
        instance.retriever_id = "atlas_rag_test"
        instance._retrieve_with_scores = MagicMock(  # type: ignore[method-assign]
            return_value=[("p1", 0.9), ("p2", 0.5), ("p3", 0.1)]
        )

        results = instance.retrieve("some question", top_k=3)

        instance._retrieve_with_scores.assert_called_once_with("some question", top_k=3)
        assert [r.chunk_id for r in results] == ["p1", "p2", "p3"]
        assert [r.rank for r in results] == [0, 1, 2]
        assert results[0].score == pytest.approx(0.9)
        assert all(r.retriever_meta == {"score_method": "hipporag"} for r in results)

    def test_satisfies_retriever_protocol_with_mocked_seam(self) -> None:
        instance = AtlasRagRetriever.__new__(AtlasRagRetriever)
        instance.retriever_id = "atlas_rag_test"
        instance._retrieve_with_scores = MagicMock(return_value=[])  # type: ignore[method-assign]
        assert isinstance(instance, Retriever)
