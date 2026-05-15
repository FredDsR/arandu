"""Retrieval backends implementing the `Retriever` Protocol (spec §4)."""

from arandu.shared.rag.retrievers.bm25 import BM25Retriever
from arandu.shared.rag.retrievers.null import NullRetriever

__all__ = ["BM25Retriever", "NullRetriever"]
