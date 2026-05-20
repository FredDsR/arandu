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
                llm_model_id="qwen3:14b",
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
                llm_model_id="qwen3:14b",
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
                llm_model_id="qwen3:14b",
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


# -- manifest integrity (Copilot review on PR #101) ----------------------


def _write_minimal_precompute(
    tmp_path: Path,
    *,
    sentence_encoder_model: str = "m",
    keyword: str = "transcriptions.json",
    include_events: bool = False,
    include_concept: bool = True,
    write_graphml_sha: bool = True,
    write_artifact_sha: bool = True,
    graphml_bytes: bytes = b"<graphml>fake</graphml>",
) -> Path:
    """Build a minimal kg_outputs_dir with manifest + (optional) sha fields.

    Used to drive the integrity-check tests without paying the embedding
    build cost. The pickles are placeholder bytes — the constructor will
    surface sha mismatches before any `pickle.load` runs.
    """
    import hashlib
    import json

    kg_outputs_dir = tmp_path / "atlas_output"
    precompute = kg_outputs_dir / "precompute"
    (kg_outputs_dir / "kg_graphml").mkdir(parents=True)
    precompute.mkdir(parents=True)

    graphml_path = kg_outputs_dir / "kg_graphml" / f"{keyword}_graph.graphml"
    graphml_path.write_bytes(graphml_bytes)

    encoder_short = sentence_encoder_model.split("/")[-1]
    flags = f"event{include_events}_concept{include_concept}"
    artifacts = {
        "node_embeddings": precompute / f"{keyword}_{flags}_{encoder_short}_node_embeddings.pkl",
        "edge_embeddings": precompute / f"{keyword}_{flags}_{encoder_short}_edge_embeddings.pkl",
        "node_list": precompute / f"{keyword}_{flags}_node_list.pkl",
        "edge_list": precompute / f"{keyword}_{flags}_edge_list.pkl",
        "text_embeddings": precompute / f"{keyword}_{encoder_short}_text_embeddings.pkl",
        "text_dict": precompute / f"{keyword}_original_text_dict_with_node_id.pkl",
    }
    # Placeholder pickle bodies; never reach `pickle.load` if sha check fires first.
    placeholder = b"placeholder-pickle-bytes"
    for path in artifacts.values():
        path.write_bytes(placeholder)

    manifest: dict[str, object] = {
        "kg_outputs_dir": str(kg_outputs_dir),
        "keyword": keyword,
        "include_events": include_events,
        "include_concept": include_concept,
        "sentence_encoder_model": sentence_encoder_model,
        "built_at": "2026-05-20T00:00:00Z",
    }
    if write_graphml_sha:
        manifest["graphml_sha256"] = hashlib.sha256(graphml_bytes).hexdigest()
    if write_artifact_sha:
        manifest["artifact_sha256"] = {
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in artifacts.items()
        }
    (precompute / "manifest.json").write_text(json.dumps(manifest))
    return kg_outputs_dir


class TestAtlasRagManifestIntegrity:
    """Drift detection: stale precompute or tampered pickles must fail loudly."""

    def test_graphml_sha_mismatch_raises(self, tmp_path: Path) -> None:
        # Manifest records a graphml sha; the on-disk graphml has different
        # bytes. The constructor must refuse to load before touching any
        # pickle (the precompute is stale relative to the rebuilt KG).
        kg_outputs_dir = _write_minimal_precompute(tmp_path)
        # Tamper with the graphml after building the manifest.
        graphml = kg_outputs_dir / "kg_graphml" / "transcriptions.json_graph.graphml"
        graphml.write_bytes(b"<graphml>DIFFERENT</graphml>")

        with pytest.raises(ValueError, match="graphml sha256 mismatch"):
            AtlasRagRetriever(
                kg_outputs_dir=kg_outputs_dir,
                llm_client=MagicMock(),
                llm_model_id="qwen3:14b",
                sentence_encoder=MagicMock(),
                sentence_encoder_model="m",
            )

    def test_manifest_without_graphml_sha_rejected(self, tmp_path: Path) -> None:
        # Hand-edited / pre-integrity-check manifests must be rejected; silent
        # loading would defeat the drift-detection guarantee.
        kg_outputs_dir = _write_minimal_precompute(tmp_path, write_graphml_sha=False)
        with pytest.raises(ValueError, match="no 'graphml_sha256'"):
            AtlasRagRetriever(
                kg_outputs_dir=kg_outputs_dir,
                llm_client=MagicMock(),
                llm_model_id="qwen3:14b",
                sentence_encoder=MagicMock(),
                sentence_encoder_model="m",
            )

    def test_tampered_pickle_raises_before_load(self, tmp_path: Path) -> None:
        # An attacker (or a partial-copy) flips bytes in one pickle but the
        # graphml is intact. sha check must fire BEFORE pickle.load — i.e.
        # we never execute the tampered payload's __reduce__.
        kg_outputs_dir = _write_minimal_precompute(tmp_path)
        # Append a byte to one pickle after the manifest was written.
        tampered = next((kg_outputs_dir / "precompute").glob("*node_embeddings.pkl"))
        tampered.write_bytes(tampered.read_bytes() + b"\x00")

        with pytest.raises(ValueError, match="sha256 mismatch"):
            AtlasRagRetriever(
                kg_outputs_dir=kg_outputs_dir,
                llm_client=MagicMock(),
                llm_model_id="qwen3:14b",
                sentence_encoder=MagicMock(),
                sentence_encoder_model="m",
            )

    def test_manifest_without_artifact_sha_rejected(self, tmp_path: Path) -> None:
        kg_outputs_dir = _write_minimal_precompute(tmp_path, write_artifact_sha=False)
        with pytest.raises(ValueError, match="no sha256 for artifact"):
            AtlasRagRetriever(
                kg_outputs_dir=kg_outputs_dir,
                llm_client=MagicMock(),
                llm_model_id="qwen3:14b",
                sentence_encoder=MagicMock(),
                sentence_encoder_model="m",
            )
