"""Tests for ``shared/embeddings/gemini.py`` — the Gemini-backed embedder.

Locks the encoder contract atlas-rag's ``HippoRAGRetriever`` calls into:

- ``encode(list[str], normalize_embeddings=bool) -> 2D ndarray``
- Empty input → ``(0, 0)`` shape without making an API call
- Batches over 100 inputs fan out into Gemini-sized sub-requests
- Extra kwargs (``query_type=...``) are swallowed without error
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np

from arandu.shared.embeddings import GeminiEmbedder

if TYPE_CHECKING:
    from collections.abc import Iterable


def _fake_embedding_response(dim: int, count: int) -> object:
    """Construct an object shaped like ``client.embeddings.create(...)``'s return.

    The real ``CreateEmbeddingResponse`` exposes ``.data`` as a list of
    ``Embedding`` objects, each carrying an ``.embedding`` list[float].
    """

    def make_emb(seed: int) -> object:
        e = MagicMock()
        # Deterministic vectors so tests can assert ordering downstream.
        e.embedding = [float(seed + i * 0.001) for i in range(dim)]
        return e

    resp = MagicMock()
    resp.data = [make_emb(seed=i) for i in range(count)]
    return resp


class TestGeminiEmbedderEncode:
    """The minimal contract atlas-rag relies on."""

    def test_empty_input_returns_zero_shape_without_calling_client(self) -> None:
        # Empty sentence list must short-circuit; atlas-rag's callers do
        # shape checks on the result, but we save a free API call regardless.
        client = MagicMock()
        embedder = GeminiEmbedder(client=client, model="gemini-embedding-001")

        result = embedder.encode([])

        assert result.shape == (0, 0)
        assert result.dtype == np.float32
        client.embeddings.create.assert_not_called()

    def test_single_batch_returns_2d_ndarray(self) -> None:
        # atlas-rag's query2edge calls `np.linalg.norm(query_emb, axis=1)`
        # so we MUST return 2D, even when input is one sentence.
        client = MagicMock()
        client.embeddings.create.return_value = _fake_embedding_response(dim=4, count=1)
        embedder = GeminiEmbedder(client=client, model="gemini-embedding-001")

        result = embedder.encode(["one sentence"])

        assert result.ndim == 2
        assert result.shape == (1, 4)

    def test_batch_chunking_fans_out_into_100_sized_subrequests(self) -> None:
        # Gemini caps each call at 100 inputs; atlas-rag hands us 256.
        # 250 inputs → 3 sub-requests (100 + 100 + 50).
        client = MagicMock()
        # Each call returns embeddings sized for what was passed in.
        client.embeddings.create.side_effect = [
            _fake_embedding_response(dim=4, count=100),
            _fake_embedding_response(dim=4, count=100),
            _fake_embedding_response(dim=4, count=50),
        ]
        embedder = GeminiEmbedder(client=client, model="gemini-embedding-001")

        sentences = [f"sentence-{i}" for i in range(250)]
        result = embedder.encode(sentences)

        assert client.embeddings.create.call_count == 3
        assert result.shape == (250, 4)

    def test_normalize_embeddings_produces_unit_rows(self) -> None:
        client = MagicMock()
        client.embeddings.create.return_value = _fake_embedding_response(dim=8, count=3)
        embedder = GeminiEmbedder(client=client, model="gemini-embedding-001")

        result = embedder.encode(["a", "b", "c"], normalize_embeddings=True)

        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_kwargs_are_swallowed(self) -> None:
        # atlas-rag's runtime path passes `query_type="edge"` and possibly
        # other forward-compatible kwargs; our encoder must accept them
        # without erroring even though we don't act on them.
        client = MagicMock()
        client.embeddings.create.return_value = _fake_embedding_response(dim=4, count=1)
        embedder = GeminiEmbedder(client=client, model="gemini-embedding-001")

        result = embedder.encode(["q"], normalize_embeddings=False, query_type="edge")
        assert result.shape == (1, 4)

    def test_dtype_is_float32(self) -> None:
        # atlas-rag's persisted pickles are float32; if we return float64
        # the downstream dot-product math gets slower + the pickles inflate.
        client = MagicMock()
        client.embeddings.create.return_value = _fake_embedding_response(dim=4, count=2)
        embedder = GeminiEmbedder(client=client, model="gemini-embedding-001")

        result = embedder.encode(["a", "b"])
        assert result.dtype == np.float32

    def test_model_attribute_is_exposed(self) -> None:
        # AtlasRagRetriever.build_index records `sentence_encoder_model` in
        # the manifest; callers read `.model` (or pass it separately). The
        # attribute is public.
        embedder = GeminiEmbedder(client=MagicMock(), model="gemini-embedding-001")
        assert embedder.model == "gemini-embedding-001"


class TestGeminiEmbedderRequestShape:
    """Wire-format expectations atlas-rag relies on."""

    def test_passes_sentences_as_input_kwarg(self) -> None:
        client = MagicMock()
        client.embeddings.create.return_value = _fake_embedding_response(dim=4, count=2)
        embedder = GeminiEmbedder(client=client, model="m")

        embedder.encode(["alpha", "beta"])

        client.embeddings.create.assert_called_once()
        _, kwargs = client.embeddings.create.call_args
        assert kwargs["model"] == "m"
        assert _equal_lists(kwargs["input"], ["alpha", "beta"])


def _equal_lists(a: Iterable[str], b: Iterable[str]) -> bool:
    return list(a) == list(b)
