"""Shared RAG primitives: Retriever Protocol and retrieval/answer record schemas."""

from arandu.shared.rag.protocol import Retriever
from arandu.shared.rag.schemas import AnswerRecord, RetrievalRecord, RetrievedPassage

__all__ = ["AnswerRecord", "RetrievalRecord", "RetrievedPassage", "Retriever"]
