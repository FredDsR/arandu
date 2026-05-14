"""Tests for shared/rag/protocol.py — Retriever Protocol structural typing."""

from __future__ import annotations

from arandu.shared.rag.protocol import Retriever
from arandu.shared.rag.schemas import RetrievedPassage


class _StubRetriever:
    """Minimal valid Retriever implementation used to exercise structural typing."""

    def __init__(self, retriever_id: str = "stub_retriever") -> None:
        self.retriever_id = retriever_id

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        # Echoes top_k so tests can assert the argument plumbing.
        return [
            RetrievedPassage(
                chunk_id=f"chunk{i:013d}",
                rank=i,
                score=1.0 - i * 0.1,
                retriever_meta={"q": question, "top_k": top_k},
            )
            for i in range(top_k)
        ]


class _MissingMethodRetriever:
    """Has retriever_id but no retrieve method."""

    retriever_id = "broken"


class TestRetrieverProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        retriever = _StubRetriever()
        assert isinstance(retriever, Retriever)

    def test_missing_retrieve_method_fails_protocol_check(self) -> None:
        broken = _MissingMethodRetriever()
        assert not isinstance(broken, Retriever)

    def test_retrieve_returns_ranked_passages(self) -> None:
        retriever = _StubRetriever()
        passages = retriever.retrieve("Em que ano?", top_k=3)
        assert len(passages) == 3
        assert [p.rank for p in passages] == [0, 1, 2]
        # Argument plumbing preserved in the stub's meta
        assert passages[0].retriever_meta["q"] == "Em que ano?"
        assert passages[0].retriever_meta["top_k"] == 3

    def test_retriever_id_exposed(self) -> None:
        retriever = _StubRetriever(retriever_id="bm25_bm25_512t")
        assert retriever.retriever_id == "bm25_bm25_512t"

    def test_protocol_accepts_class_with_callable_retrieve(self) -> None:
        # A different class shape (e.g. a lambda assigned to retrieve) should still satisfy.
        class _Lambdic:
            retriever_id = "lambdic"

            def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
                return []

        assert isinstance(_Lambdic(), Retriever)
