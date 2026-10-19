"""Retrieval backends implementing the `Retriever` Protocol (spec §4)."""

from arandu.shared.rag.retrievers.atlas_rag import AtlasRagRetriever
from arandu.shared.rag.retrievers.bm25 import BM25Retriever
from arandu.shared.rag.retrievers.khop_subgraph import KHopSubgraphRetriever
from arandu.shared.rag.retrievers.khop_triple import KHopTripleRetriever
from arandu.shared.rag.retrievers.null import NullRetriever
from arandu.shared.rag.retrievers.oracle import OracleRetriever

__all__ = [
    "AtlasRagRetriever",
    "BM25Retriever",
    "KHopSubgraphRetriever",
    "KHopTripleRetriever",
    "NullRetriever",
    "OracleRetriever",
]
