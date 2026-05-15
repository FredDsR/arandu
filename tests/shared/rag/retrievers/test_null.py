"""Tests for shared/rag/retrievers/null.py — NullRetriever parametric arm."""

from __future__ import annotations

from arandu.shared.rag.protocol import Retriever
from arandu.shared.rag.retrievers.null import NullRetriever


class TestNullRetriever:
    def test_retriever_id_is_null(self) -> None:
        assert NullRetriever().retriever_id == "null"

    def test_satisfies_retriever_protocol(self) -> None:
        assert isinstance(NullRetriever(), Retriever)

    def test_retrieve_returns_empty_list(self) -> None:
        retriever = NullRetriever()
        assert retriever.retrieve("Em que ano ocorreu a enchente?", top_k=10) == []

    def test_retrieve_returns_empty_regardless_of_top_k(self) -> None:
        retriever = NullRetriever()
        for k in (1, 5, 100):
            assert retriever.retrieve("qualquer pergunta", top_k=k) == []

    def test_retrieve_returns_empty_for_empty_question(self) -> None:
        # Spec §4.6 says trivial implementation — input is not validated;
        # the Answerer downstream simply receives no passages.
        assert NullRetriever().retrieve("", top_k=5) == []
