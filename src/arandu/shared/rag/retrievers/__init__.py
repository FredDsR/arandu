"""Retrieval backends implementing the `Retriever` Protocol (spec §4)."""

from arandu.shared.rag.retrievers.null import NullRetriever

__all__ = ["NullRetriever"]
