"""Gemini-backed embedder for atlas-rag's encoder interface.

Promoted from ``scripts/test_atlas_rag_retriever.py`` after the smoke run
on ``test-kg-04`` (2026-05-22) confirmed the throttling + batch shape
hold up under a real KG build (~75k embeddings, ~5 min wall time, ~$0.05).

atlas-rag's ``HippoRAGRetriever`` calls ``sentence_encoder.encode(batch,
normalize_embeddings=bool)`` and nothing else from the encoder, so this
adapter duck-types against atlas-rag's :class:`BaseEmbeddingModel`
rather than subclassing it — avoids a hard import dependency on the
atlas-rag class hierarchy at module-load time.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openai import OpenAI


# Gemini's BatchEmbedContentsRequest caps each call at 100 inputs. atlas-rag
# hands us batches of up to 256, so we fan each call out into Gemini-sized
# sub-requests.
_GEMINI_BATCH_MAX = 100

# Gemini's paid-tier embed quota is ~3000 RPM. 50ms between sub-requests
# = 20 RPS = 1200 RPM, well under the ceiling with headroom for retries.
_PACING_SLEEP_S = 0.05


class GeminiEmbedder:
    """Adapter exposing Gemini embeddings via atlas-rag's encoder interface.

    Uses Gemini's OpenAI-compatible endpoint so the same ``OpenAI`` client
    works for both LLM completions and embeddings — no new SDK in the
    dependency tree.

    Two layers of throttle handling for Gemini's quotas:

    - **Batch size cap.** Gemini's BatchEmbedContentsRequest limits each call
      to 100 inputs; atlas-rag hands us batches of 256. We fan each call out
      into Gemini-sized sub-requests.
    - **Per-minute pacing + retry.** Pacing constant (``_PACING_SLEEP_S``)
      keeps RPS well under the 3000 RPM ceiling; the ``openai`` SDK's
      built-in ``Retry-After`` honouring (via ``max_retries``) handles
      stragglers.

    Attributes:
        model: The Gemini embedding model identifier (e.g.
            ``"gemini-embedding-001"``).
    """

    def __init__(self, client: OpenAI, model: str) -> None:
        """Construct from an existing OpenAI-shaped client + model id.

        Args:
            client: An ``openai.OpenAI`` instance pointed at Gemini's
                ``/v1beta/openai/`` base URL.
            model: Embedding model identifier (e.g. ``gemini-embedding-001``).
                The 004 series was removed by Google in 2025-Q1; current code
                uses 001 unless overridden.
        """
        self._client = client
        self.model = model

    def encode(
        self,
        sentences: list[str],
        normalize_embeddings: bool = False,
        **_kwargs: object,  # atlas-rag passes `query_type` etc.; ignore.
    ) -> np.ndarray:
        """Encode sentences into a 2D ``(batch, dim)`` numpy array.

        atlas-rag's ``HippoRAGRetriever.query2edge`` calls
        ``np.linalg.norm(query_emb, axis=1, keepdims=True)`` and
        ``edge_embeddings @ query_emb[0].T``, both of which require a 2D
        ndarray. ``list[ndarray]`` works for the build-time ``.extend()``
        pattern but breaks query-time math — so we always return 2D.

        Args:
            sentences: Strings to embed.
            normalize_embeddings: If true, L2-normalize each row of the
                result (atlas-rag uses this for cosine similarity).
            **_kwargs: Swallowed. atlas-rag's runtime path passes
                ``query_type`` and possibly other forward-compatible
                kwargs that aren't part of our minimal contract.

        Returns:
            A ``(len(sentences), embedding_dim)`` float32 ndarray. An
            empty input returns a ``(0, 0)`` ndarray (atlas-rag's
            callers handle the empty case via shape checks).
        """
        if not sentences:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for start in range(0, len(sentences), _GEMINI_BATCH_MAX):
            sub = sentences[start : start + _GEMINI_BATCH_MAX]
            resp = self._client.embeddings.create(model=self.model, input=sub)
            vectors.extend(np.array(d.embedding, dtype=np.float32) for d in resp.data)
            if start + _GEMINI_BATCH_MAX < len(sentences):
                time.sleep(_PACING_SLEEP_S)
        arr = np.stack(vectors).astype(np.float32)
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr
