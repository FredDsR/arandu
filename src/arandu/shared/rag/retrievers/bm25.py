"""BM25 sparse retrieval baseline (spec §4.3).

Uses ``rank_bm25.BM25Okapi``. Tokenization is delegated to
:mod:`arandu.shared.rag.retrievers._bm25_tokenize` (spaCy primary with a
whitespace fallback). Index persistence is a pickled ``BM25Okapi`` plus a
JSON manifest. The manifest carries a SHA-256 of the pickle bytes, verified
before ``pickle.load`` to detect corruption / tampering of the index file.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from arandu.shared.rag.retrievers._bm25_tokenize import (
    english_tokenizer,
    portuguese_tokenizer,
)
from arandu.shared.rag.schemas import RetrievedPassage

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from arandu.shared.chunking.resolver import ChunkResolver
    from arandu.shared.chunking.schemas import Chunk


INDEX_FILENAME = "bm25.pkl"
MANIFEST_FILENAME = "manifest.json"
RETRIEVER_FAMILY = "bm25"


def _make_tokenizer(language: str) -> Callable[[str], list[str]]:
    """Return a tokenizer callable for ``language``.

    Raises:
        ValueError: If ``language`` is neither ``"pt"`` nor ``"en"``.
    """
    if language == "pt":
        return portuguese_tokenizer()
    if language == "en":
        return english_tokenizer()
    raise ValueError(f"Unsupported language for BM25: {language!r}. Use 'pt' or 'en'.")


def _derive_retriever_id(chunker_id: str) -> str:
    """Compose the retriever_id, avoiding a redundant ``bm25_`` family prefix.

    Chunker IDs in this codebase typically already prefix their intended
    retriever family (e.g. ``bm25_512t``); concatenating ``bm25_`` again would
    produce ``bm25_bm25_512t``. When the family prefix is already present,
    return the chunker_id verbatim; otherwise prepend ``bm25_`` so that a
    non-family chunker view (e.g. ``cep_4k``) is still distinguished from
    other retrievers indexing the same view.
    """
    prefix = f"{RETRIEVER_FAMILY}_"
    return chunker_id if chunker_id.startswith(prefix) else f"{prefix}{chunker_id}"


class BM25Retriever:
    """BM25 retriever over a chunker view.

    The retriever loads a pre-built index from ``index_dir`` (produced by
    :meth:`build_index`). The ``retriever_id`` is derived from ``chunker_id``
    by :func:`_derive_retriever_id`: when ``chunker_id`` already starts with
    ``bm25_`` (the chunk-size-sweep convention — ``bm25_512t``, ``bm25_1024t``,
    ``bm25_4k``), the chunker_id is used verbatim. Otherwise the family prefix
    is added so a non-family chunker view (e.g. ``cep_4k``) yields a
    distinguishable ``bm25_cep_4k`` and does not collide with another
    retriever indexing the same view (e.g. ``networkx_cep_4k``).
    """

    retriever_id: str

    def __init__(
        self,
        index_dir: Path,
        chunker_id: str,
        language: str = "pt",
    ) -> None:
        """Load a BM25 index from ``index_dir``.

        Args:
            index_dir: Directory containing ``bm25.pkl`` and ``manifest.json``
                (typically ``retrieval_indexes/<chunker_id>/bm25/``).
            chunker_id: Chunker view ID this index was built from. Must match
                the value recorded in the manifest; mismatch raises ``ValueError``.
            language: ``"pt"`` or ``"en"``. Must match the language used at
                index-build time (recorded in the manifest); mismatch would
                silently destroy ranking quality, so it raises ``ValueError``.

        Raises:
            FileNotFoundError: If ``index_dir`` lacks either index file.
            ValueError: If ``language`` is unsupported, the manifest disagrees
                with the constructor args, or the recorded SHA-256 does not
                match the pickle bytes on disk.
        """
        self.retriever_id = _derive_retriever_id(chunker_id)
        self._chunker_id = chunker_id
        self._language = language
        self._tokenize = _make_tokenizer(language)
        self._bm25, self._chunk_ids = self._load_index(
            index_dir, expected_chunker_id=chunker_id, expected_language=language
        )

    @classmethod
    def build_index(
        cls,
        chunks: list[Chunk],
        resolver: ChunkResolver,
        index_dir: Path,
        chunker_id: str,
        language: str = "pt",
    ) -> None:
        """Build a BM25 index from a chunk list and persist it to ``index_dir``.

        Args:
            chunks: Ordered chunks to index; resolved via ``resolver``.
            resolver: Resolves chunks to source text.
            index_dir: Output directory; created if missing.
            chunker_id: Chunker view ID; recorded in the manifest.
            language: Tokenizer language (``"pt"`` or ``"en"``).

        Raises:
            ValueError: If ``chunks`` is empty or ``language`` is unsupported.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from an empty chunks list.")
        tokenize = _make_tokenizer(language)
        tokenized = [tokenize(resolver.text(c)) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        chunk_ids = [c.chunk_id for c in chunks]

        index_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = index_dir / INDEX_FILENAME
        with pkl_path.open("wb") as f:
            pickle.dump(bm25, f)
        sha256 = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
        manifest = {
            "chunker_id": chunker_id,
            "language": language,
            "chunks_indexed": len(chunks),
            "built_at": datetime.now(UTC).isoformat(),
            "sha256": sha256,
            "chunk_ids": chunk_ids,
        }
        (index_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def _load_index(
        index_dir: Path,
        expected_chunker_id: str,
        expected_language: str,
    ) -> tuple[BM25Okapi, list[str]]:
        idx_path = index_dir / INDEX_FILENAME
        mfst_path = index_dir / MANIFEST_FILENAME
        if not idx_path.exists() or not mfst_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_dir} "
                f"(expected '{INDEX_FILENAME}' and '{MANIFEST_FILENAME}')."
            )
        manifest = json.loads(mfst_path.read_text())
        if manifest["chunker_id"] != expected_chunker_id:
            raise ValueError(
                f"chunker_id mismatch: index at {index_dir} was built for "
                f"{manifest['chunker_id']!r}, but constructor received "
                f"{expected_chunker_id!r}."
            )
        if manifest["language"] != expected_language:
            raise ValueError(
                f"language mismatch: index at {index_dir} was tokenized with "
                f"{manifest['language']!r}, but constructor received "
                f"{expected_language!r}. Loading with a different language would "
                f"silently destroy ranking quality."
            )
        recorded_sha = manifest.get("sha256")
        if recorded_sha is None:
            raise ValueError(
                f"manifest at {mfst_path} has no 'sha256' field — refusing to "
                f"load an unverifiable BM25 index. Rebuild the index."
            )
        actual_sha = hashlib.sha256(idx_path.read_bytes()).hexdigest()
        if actual_sha != recorded_sha:
            raise ValueError(
                f"sha256 mismatch on {idx_path}: manifest records "
                f"{recorded_sha[:12]}…, computed {actual_sha[:12]}… — index may "
                f"be corrupted or tampered. Rebuild the index."
            )
        with idx_path.open("rb") as f:
            bm25 = pickle.load(f)
        return bm25, manifest["chunk_ids"]

    def retrieve(self, question: str, top_k: int) -> list[RetrievedPassage]:
        """Retrieve up to ``top_k`` chunks ranked by BM25 score.

        Args:
            question: Natural-language query; tokenized with the same
                language tokenizer used at index build time.
            top_k: Maximum number of passages to return.

        Returns:
            A list of :class:`RetrievedPassage` with consecutive
            zero-indexed ranks and monotonically non-increasing scores.
        """
        tokens = self._tokenize(question)
        scores = self._bm25.get_scores(tokens)
        ranked_idx = (-scores).argsort(kind="stable")[:top_k]
        return [
            RetrievedPassage(
                chunk_id=self._chunk_ids[int(i)],
                rank=r,
                score=float(scores[int(i)]),
                retriever_meta={"score_method": "bm25_okapi"},
            )
            for r, i in enumerate(ranked_idx)
        ]
