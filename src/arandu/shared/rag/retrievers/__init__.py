"""Retrieval backends implementing the `Retriever` Protocol (spec §4)."""

from arandu.shared.rag.retrievers.atlas_rag import AtlasRagRetriever
from arandu.shared.rag.retrievers.bm25 import BM25Retriever
from arandu.shared.rag.retrievers.khop_subgraph import KHopSubgraphRetriever
from arandu.shared.rag.retrievers.networkx_triple import NetworkXTripleRetriever
from arandu.shared.rag.retrievers.null import NullRetriever

__all__ = [
    "AtlasRagRetriever",
    "BM25Retriever",
    "KHopSubgraphRetriever",
    "NetworkXTripleRetriever",
    "NullRetriever",
]
