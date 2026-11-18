"""Dense (embedding-based) retrieval baseline over a chunker view.

The arm indexes the same ``cep_4k`` chunks as BM25 but scores them with a
sentence encoder. A 4,000-character chunk is longer than the input of most
sentence encoders (128 tokens for ``paraphrase-multilingual-mpnet-base-v2``),
which would silently truncate it, so each chunk is split into overlapping
token windows (sub-passages) cut with the encoder's own tokenizer. Encoders
with a long input (``BAAI/bge-m3``, ``Qwen/Qwen3-Embedding-0.6B``) can also
read each chunk whole, as one sub-passage. The sub-passages and the question
are embedded with L2 normalization, and a sub-passage scores by cosine
similarity (a dot product of unit vectors).

Instruction-tuned encoders expect a prompt in front of the query and,
sometimes, of the passage (``multilingual-e5-large-instruct`` and
Qwen3-Embedding take an ``Instruct: ...`` query prefix and no passage
prefix). The encoder carries both prompts; the passage prompt is part of
the index and recorded in its manifest.

By default the arm returns parent chunks: a chunk scores the maximum
similarity of its sub-passages, and the top ``k`` chunks are returned with
the same ``chunk_id`` as BM25. The evidence unit and the amount of text that
reaches the answerer stay comparable with BM25 and the oracle, and the rank
of a pair's source chunk is measured the same way for every chunk-level arm.
The ``subpassage`` unit returns the windows themselves, carried inline.

The index is a plain normalized ``float32`` matrix searched by brute force
(a few thousand rows), persisted as ``embeddings.npy`` + ``subpassages.json``
+ a JSON manifest that records the encoder, the window, the passage prompt
and a SHA-256 of both data files, verified before loading.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from arandu.shared.rag.schemas import RetrievedPassage

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from arandu.shared.chunking.resolver import ChunkResolver
    from arandu.shared.chunking.schemas import Chunk
    from arandu.shared.rag.schemas import RetrievalQuery

logger = logging.getLogger(__name__)

EMBEDDINGS_FILENAME = "embeddings.npy"
SUBPASSAGES_FILENAME = "subpassages.json"
MANIFEST_FILENAME = "manifest.json"
RETRIEVER_FAMILY = "dense"
SCORE_METHOD = "cosine_max_subpassage"
WHOLE_CHUNK = "whole"

DenseUnit = Literal["parent", "subpassage"]


class DenseEncoder(Protocol):
    """What the dense arm needs from a sentence encoder.

    Attributes:
        model_name: Identifier recorded in the index manifest.
        max_seq_length: Longest input, in tokens and including special
            tokens and prompt, the encoder reads before truncating.
        num_special_tokens: Special tokens the encoder adds to one input
            (``<s>`` and ``</s>`` for mpnet).
        query_prompt: Text prepended to every query (``""`` for none).
        passage_prompt: Text prepended to every passage (``""`` for none).
    """

    model_name: str
    max_seq_length: int
    num_special_tokens: int
    query_prompt: str
    passage_prompt: str

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Return the ``(start_char, end_char)`` span of each token of ``text``."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return how many tokens the encoder reads for ``text``, special tokens excluded."""
        ...

    def encode(self, texts: list[str], *, query: bool) -> np.ndarray:
        """Return one L2-normalized ``float32`` row per text, with the matching prompt."""
        ...


class SentenceTransformerEncoder:
    """:class:`DenseEncoder` over a local ``sentence_transformers`` model.

    ``sentence_transformers`` ships with the ``kg`` extra (it is the encoder
    atlas-rag uses), so it is imported lazily. When a CUDA forward pass runs
    out of memory (the GPU may be shared with an Ollama server), the model
    moves to the CPU and the batch is retried there.
    """

    def __init__(
        self,
        model_name: str,
        *,
        query_prompt: str = "",
        passage_prompt: str = "",
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        """Load ``model_name`` and its tokenizer.

        Args:
            model_name: Any ``SentenceTransformer``-resolvable name.
            query_prompt: Text prepended to queries.
            passage_prompt: Text prepended to passages.
            device: Torch device (``"cpu"``, ``"cuda"``); ``None`` lets
                sentence-transformers pick.
            batch_size: Texts per forward pass when encoding.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prompt = query_prompt
        self.passage_prompt = passage_prompt
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _is_cuda_memory_error(exc):
                raise
            logger.warning("CUDA out of memory loading %s; loading on CPU.", model_name)
            torch.cuda.empty_cache()
            self._model = SentenceTransformer(model_name, device="cpu")
        self._tokenizer = self._model.tokenizer
        self._batch_size = batch_size
        self.max_seq_length = int(self._model.max_seq_length)
        self.num_special_tokens = int(self._tokenizer.num_special_tokens_to_add(pair=False))

    @property
    def device(self) -> str:
        """Device the model currently runs on."""
        return str(self._model.device)

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Return the character span of every token of ``text`` (no special tokens)."""
        enc = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        return [(int(s), int(e)) for s, e in enc["offset_mapping"] if e > s]

    def count_tokens(self, text: str) -> int:
        """Return the tokens the encoder reads for ``text``, special tokens excluded.

        Unlike :meth:`token_spans`, zero-width tokens (a lone word-start
        marker) count, since they take a place in the encoder input.
        """
        return len(
            self._tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)[
                "input_ids"
            ]
        )

    def encode(self, texts: list[str], *, query: bool) -> np.ndarray:
        """Encode ``texts`` into L2-normalized ``float32`` rows."""
        import torch

        prompt = (self.query_prompt if query else self.passage_prompt) or None
        try:
            emb = self._encode(texts, prompt)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if self.device == "cpu" or not _is_cuda_memory_error(exc):
                raise
            logger.warning("CUDA out of memory encoding with %s; moving to CPU.", self.model_name)
            self._model.to("cpu")
            torch.cuda.empty_cache()
            emb = self._encode(texts, prompt)
        return np.asarray(emb, dtype=np.float32)

    def _encode(self, texts: list[str], prompt: str | None) -> np.ndarray:
        return self._model.encode(
            texts,
            prompt=prompt,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def _is_cuda_memory_error(exc: BaseException) -> bool:
    """Whether ``exc`` is CUDA running out of memory (also as a cuBLAS allocation failure)."""
    text = str(exc)
    return type(exc).__name__ == "OutOfMemoryError" or (
        "CUDA" in text and ("out of memory" in text or "ALLOC_FAILED" in text)
    )


def resolve_window(
    encoder: DenseEncoder, window_tokens: int | None, overlap_ratio: float
) -> tuple[int, int]:
    """Resolve the sub-passage window into content tokens per window and overlap.

    ``window_tokens`` is the encoder input one sub-passage fills, special
    tokens and passage prompt included, so ``128`` means a full mpnet input.
    The content of a window is what remains after those.

    Args:
        encoder: The encoder whose input limit bounds the window.
        window_tokens: Input tokens per sub-passage; ``None`` takes the
            encoder's ``max_seq_length``.
        overlap_ratio: Share of a window's content repeated by the next window.

    Returns:
        ``(content_tokens, overlap_tokens)`` with ``0 <= overlap < content``.

    Raises:
        ValueError: If the window exceeds the encoder's input, leaves no
            room for content, or the ratio is outside ``[0, 1)``.
    """
    window = encoder.max_seq_length if window_tokens is None else window_tokens
    if window > encoder.max_seq_length:
        raise ValueError(
            f"window_tokens={window} exceeds what {encoder.model_name!r} reads "
            f"({encoder.max_seq_length} tokens); longer windows would be truncated."
        )
    prompt_tokens = len(encoder.token_spans(encoder.passage_prompt))
    content = window - encoder.num_special_tokens - prompt_tokens
    if content < 1:
        raise ValueError(
            f"window_tokens={window} leaves no content after {encoder.num_special_tokens} "
            f"special and {prompt_tokens} prompt tokens."
        )
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}.")
    return content, min(round(content * overlap_ratio), content - 1)


def split_windows(
    token_spans: list[tuple[int, int]], window: int, overlap: int
) -> list[tuple[int, int]]:
    """Cut a token sequence into overlapping windows, as character spans.

    Args:
        token_spans: ``(start_char, end_char)`` per token, in text order.
        window: Tokens per window.
        overlap: Tokens shared by consecutive windows (``< window``).

    Returns:
        ``(start_char, end_char)`` per window. The last window ends at the
        last token; an empty token list yields ``[]``.

    Examples:
        >>> split_windows([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)], 2, 1)
        [(0, 3), (2, 5), (4, 7), (6, 9)]
    """
    n = len(token_spans)
    if n == 0:
        return []
    stride = window - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + window, n)
        spans.append((token_spans[start][0], token_spans[end - 1][1]))
        if end == n:
            return spans
        start += stride


def fit_windows(
    text: str,
    token_spans: list[tuple[int, int]],
    window: int,
    overlap: int,
    count_tokens: Callable[[str], int],
) -> list[tuple[int, int]]:
    """Cut ``text`` into overlapping windows that each fit ``window`` tokens when re-tokenized.

    The windows are cut on the tokens of the whole chunk, but the encoder
    tokenizes each window's text on its own. A window that starts or ends
    inside a word can come out longer (a word piece gains a word-start
    marker), and the encoder would then drop its last tokens. Each window
    is therefore checked with ``count_tokens`` and shortened from the end
    until it fits; the next window starts ``overlap`` tokens before the
    shortened end, so every token stays covered. With no shortening the
    windows are those of :func:`split_windows`.

    Args:
        text: The chunk text the spans index into.
        token_spans: ``(start_char, end_char)`` per token of ``text``.
        window: Content tokens per window.
        overlap: Tokens shared by consecutive windows (``< window``).
        count_tokens: Tokens the encoder reads for a string.

    Returns:
        ``(start_char, end_char)`` per window; ``[]`` for no tokens.
    """
    n = len(token_spans)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + window, n)
        while (
            end - start > 1
            and count_tokens(text[token_spans[start][0] : token_spans[end - 1][1]]) > window
        ):
            end -= 1
        spans.append((token_spans[start][0], token_spans[end - 1][1]))
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return spans


def _derive_retriever_id(chunker_id: str) -> str:
    """``dense_<chunker_id>``, mirroring BM25's ``bm25_<chunker_id>``."""
    prefix = f"{RETRIEVER_FAMILY}_"
    return chunker_id if chunker_id.startswith(prefix) else f"{prefix}{chunker_id}"


def _sha256(*paths: Path) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()


def _index_shape(
    encoder: DenseEncoder, window_tokens: int | None, whole_chunk: bool, overlap_ratio: float
) -> dict[str, object]:
    """The manifest fields that identify how the sub-passages were cut.

    ``windowing`` names the cutting rule: ``"fit"`` for :func:`fit_windows`
    (windows checked against the re-tokenized length). An index built
    before that rule has no such field and must be rebuilt.
    """
    if whole_chunk:
        return {"window_tokens": WHOLE_CHUNK, "content_tokens": None, "overlap_tokens": 0}
    content, overlap = resolve_window(encoder, window_tokens, overlap_ratio)
    window = encoder.max_seq_length if window_tokens is None else window_tokens
    return {
        "window_tokens": window,
        "content_tokens": content,
        "overlap_tokens": overlap,
        "windowing": "fit",
    }


class DenseRetriever:
    """Dense retriever over a chunker view, scoring chunks by their best sub-passage.

    Attributes:
        retriever_id: ``dense_<chunker_id>`` by default (e.g. ``dense_cep_4k``);
            named configurations pass their own.
        last_retrieve_meta: Record-level diagnostics the batch runner copies
            into each ``RetrievalRecord`` (encoder, window, prompts, unit).
    """

    retriever_id: str

    def __init__(
        self,
        index_dir: Path,
        encoder: DenseEncoder,
        chunker_id: str,
        *,
        window_tokens: int | None = None,
        whole_chunk: bool = False,
        overlap_ratio: float = 0.25,
        unit: DenseUnit = "parent",
        retriever_id: str | None = None,
    ) -> None:
        """Load a dense index and check it matches the requested configuration.

        Args:
            index_dir: Directory written by :meth:`build_index`.
            encoder: Encoder for the questions; must be the one that built
                the index (checked against the manifest).
            chunker_id: Chunker view the index must have been built from.
            window_tokens: Encoder input per sub-passage the index must
                have; ``None`` means the encoder's full input.
            whole_chunk: Whether the index must hold each chunk whole.
            overlap_ratio: Overlap the index must have, as a share of the window.
            unit: ``"parent"`` returns chunks scored by their best
                sub-passage; ``"subpassage"`` returns the windows themselves.
            retriever_id: Identifier for the records; defaults to
                ``dense_<chunker_id>``.

        Raises:
            FileNotFoundError: If an index file is missing.
            ValueError: If the manifest disagrees with the arguments or a
                recorded SHA-256 does not match the files on disk.
        """
        if unit not in ("parent", "subpassage"):
            raise ValueError(f"unit must be 'parent' or 'subpassage', got {unit!r}.")
        shape = _index_shape(encoder, window_tokens, whole_chunk, overlap_ratio)
        self.retriever_id = retriever_id or _derive_retriever_id(chunker_id)
        self._encoder = encoder
        self._unit: DenseUnit = unit
        manifest, self._emb, subs = self._load_index(
            index_dir,
            expected={
                "chunker_id": chunker_id,
                "encoder_model": encoder.model_name,
                "passage_prompt": encoder.passage_prompt,
                **shape,
            },
        )
        self._sub_chunk_ids = [str(s["chunk_id"]) for s in subs]
        self._sub_text = [str(s["text"]) for s in subs]
        self._sub_index_in_chunk = [int(s["index_in_chunk"]) for s in subs]
        self.last_retrieve_meta: dict[str, object] = {
            "encoder_model": encoder.model_name,
            "query_prompt": encoder.query_prompt,
            "passage_prompt": encoder.passage_prompt,
            **shape,
            "unit": unit,
            "subpassages_indexed": manifest["subpassages_indexed"],
        }

    @classmethod
    def build_index(
        cls,
        chunks: list[Chunk],
        resolver: ChunkResolver,
        index_dir: Path,
        chunker_id: str,
        encoder: DenseEncoder,
        *,
        window_tokens: int | None = None,
        whole_chunk: bool = False,
        overlap_ratio: float = 0.25,
    ) -> None:
        """Split, embed and persist the sub-passages of ``chunks``.

        Args:
            chunks: Ordered chunks to index; resolved via ``resolver``.
            resolver: Resolves chunks to source text.
            index_dir: Output directory; created if missing.
            chunker_id: Chunker view ID; recorded in the manifest.
            encoder: Encoder whose tokenizer cuts the windows and whose
                embeddings fill the matrix.
            window_tokens: Encoder input per sub-passage, special tokens
                and passage prompt included; ``None`` means the encoder's
                full input. Ignored when ``whole_chunk`` is set.
            whole_chunk: Encode each chunk whole, as one sub-passage. Chunks
                longer than the encoder's input are truncated by it and
                counted in the manifest.
            overlap_ratio: Share of a window repeated by the next window.

        Raises:
            ValueError: If ``chunks`` is empty, no chunk has text, or the
                window is invalid for ``encoder``.
        """
        if not chunks:
            raise ValueError("Cannot build a dense index from an empty chunks list.")
        shape = _index_shape(encoder, window_tokens, whole_chunk, overlap_ratio)
        prompt_tokens = len(encoder.token_spans(encoder.passage_prompt))
        subs: list[dict[str, object]] = []
        empty_chunks = truncated = 0
        for chunk in chunks:
            text = resolver.text(chunk)
            tokens = encoder.token_spans(text)
            if whole_chunk:
                spans = [(tokens[0][0], tokens[-1][1])] if tokens else []
                used = len(tokens) + encoder.num_special_tokens + prompt_tokens
                truncated += used > encoder.max_seq_length
            else:
                spans = fit_windows(
                    text,
                    tokens,
                    int(shape["content_tokens"]),
                    int(shape["overlap_tokens"]),
                    encoder.count_tokens,
                )
            if not spans:
                empty_chunks += 1
            for i, (start, end) in enumerate(spans):
                subs.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "index_in_chunk": i,
                        "start_char": start,
                        "end_char": end,
                        "text": text[start:end],
                    }
                )
        if not subs:
            raise ValueError("No chunk produced a sub-passage; nothing to index.")
        if empty_chunks:
            logger.warning("%d chunk(s) have no tokens and cannot be retrieved.", empty_chunks)
        if truncated:
            logger.warning(
                "%d whole chunk(s) exceed the encoder input and are truncated.", truncated
            )
        emb = encoder.encode([str(s["text"]) for s in subs], query=False).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms == 0, 1.0, norms)

        index_dir.mkdir(parents=True, exist_ok=True)
        emb_path = index_dir / EMBEDDINGS_FILENAME
        subs_path = index_dir / SUBPASSAGES_FILENAME
        np.save(emb_path, emb, allow_pickle=False)
        subs_path.write_text(json.dumps(subs, ensure_ascii=False))
        manifest = {
            "chunker_id": chunker_id,
            "encoder_model": encoder.model_name,
            "encoder_max_seq_length": encoder.max_seq_length,
            "passage_prompt": encoder.passage_prompt,
            **shape,
            "overlap_ratio": 0.0 if whole_chunk else overlap_ratio,
            "similarity": "cosine (dot product of L2-normalized embeddings)",
            "chunks_indexed": len(chunks),
            "chunks_without_subpassages": empty_chunks,
            "chunks_truncated": truncated,
            "subpassages_indexed": len(subs),
            "embedding_dim": int(emb.shape[1]),
            "built_at": datetime.now(UTC).isoformat(),
            "sha256": _sha256(emb_path, subs_path),
            "chunk_ids": [c.chunk_id for c in chunks],
        }
        (index_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def _load_index(
        index_dir: Path, expected: dict[str, object]
    ) -> tuple[dict[str, object], np.ndarray, list[dict[str, object]]]:
        emb_path = index_dir / EMBEDDINGS_FILENAME
        subs_path = index_dir / SUBPASSAGES_FILENAME
        mfst_path = index_dir / MANIFEST_FILENAME
        missing = [p.name for p in (emb_path, subs_path, mfst_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Dense index at {index_dir} is missing {missing}.")
        manifest = json.loads(mfst_path.read_text())
        for key, want in expected.items():
            if manifest.get(key) != want:
                raise ValueError(
                    f"{key} mismatch: dense index at {index_dir} has {manifest.get(key)!r}, "
                    f"the configuration asks for {want!r}. Rebuild with --rebuild-index."
                )
        actual = _sha256(emb_path, subs_path)
        if actual != manifest.get("sha256"):
            raise ValueError(
                f"sha256 mismatch on the dense index at {index_dir}: the files may be "
                f"corrupted or edited. Rebuild with --rebuild-index."
            )
        emb = np.load(emb_path, allow_pickle=False)
        subs = json.loads(subs_path.read_text())
        if emb.shape[0] != len(subs):
            raise ValueError(
                f"Dense index at {index_dir} has {emb.shape[0]} embeddings "
                f"for {len(subs)} sub-passages."
            )
        return manifest, emb, subs

    def retrieve(self, query: RetrievalQuery, top_k: int) -> list[RetrievedPassage]:
        """Retrieve up to ``top_k`` units ranked by cosine similarity.

        Args:
            query: The item to retrieve for; this arm reads only ``query.question``.
            top_k: Maximum passages; ``0`` yields ``[]``.

        Returns:
            Passages with consecutive zero-indexed ranks and non-increasing
            scores. In the ``parent`` unit each passage is a chunk whose
            ``retriever_meta`` names its best sub-passage; in the
            ``subpassage`` unit each passage carries its window inline.
        """
        if top_k < 1:
            return []
        return self.retrieve_by_embedding(
            self._encoder.encode([query.question], query=True)[0], top_k
        )

    def retrieve_by_embedding(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[RetrievedPassage]:
        """Rank the index against an already encoded, normalized query.

        Lets a caller encode many questions in one batch (analysis scripts)
        and get exactly what :meth:`retrieve` returns.

        Args:
            query_embedding: One L2-normalized row from the same encoder,
                with its query prompt.
            top_k: Maximum passages; ``0`` yields ``[]``.

        Returns:
            The same passages :meth:`retrieve` would return.
        """
        if top_k < 1:
            return []
        sims = self._emb @ query_embedding
        order = np.argsort(-sims, kind="stable")
        if self._unit == "subpassage":
            top = order[:top_k]
            return [self._subpassage(int(i), r, float(sims[i])) for r, i in enumerate(top)]
        passages: list[RetrievedPassage] = []
        seen: set[str] = set()
        for i in order:
            chunk_id = self._sub_chunk_ids[int(i)]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            score = float(sims[i])
            passages.append(
                RetrievedPassage(
                    chunk_id=chunk_id,
                    rank=len(passages),
                    score=score,
                    retriever_meta={
                        "score_method": SCORE_METHOD,
                        "best_subpassage": self._sub_index_in_chunk[int(i)],
                        "best_subpassage_score": score,
                    },
                )
            )
            if len(passages) == top_k:
                break
        return passages

    def _subpassage(self, i: int, rank: int, score: float) -> RetrievedPassage:
        index_in_chunk = self._sub_index_in_chunk[i]
        parent = self._sub_chunk_ids[i]
        return RetrievedPassage(
            chunk_id=f"{parent}:sp{index_in_chunk}",
            rank=rank,
            score=score,
            retriever_meta={
                "score_method": "cosine",
                "parent_chunk_id": parent,
                "subpassage": index_in_chunk,
            },
            payload=self._sub_text[i],
            payload_is_prose=True,
        )
