"""Embedding providers for atlas-rag retriever precompute (and beyond).

Two providers ship today: Gemini (cloud, default for thesis runs) and
sentence-transformers (local GPU, used on PCAD). See
:mod:`arandu.shared.embeddings.factory` for the selection mechanism.
"""

from arandu.shared.embeddings.factory import (
    EmbedderProvider,
    EmbedderSettings,
    build_embedder,
)
from arandu.shared.embeddings.gemini import GeminiEmbedder

__all__ = ["EmbedderProvider", "EmbedderSettings", "GeminiEmbedder", "build_embedder"]
