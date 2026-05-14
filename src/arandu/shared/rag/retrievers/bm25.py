"""BM25 sparse retrieval baseline (spec §4.3).

Uses ``rank_bm25.BM25Okapi``. Tokenization is delegated to
:mod:`arandu.shared.rag.retrievers._bm25_tokenize` (spaCy primary with a
whitespace fallback). Index persistence is a pickled ``BM25Okapi`` plus a
JSON manifest of ``chunk_ids``.
"""

from __future__ import annotations

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


class BM25Retriever:
    """BM25 retriever over a chunker view.

    The retriever loads a pre-built index from ``index_dir`` (produced by
    :meth:`build_index`). The ``retriever_id`` is ``f"bm25_{chunker_id}"``;
    different chunk-size sweeps (``bm25_512t``, ``bm25_1024t``, ``bm25_4k``)
    therefore produce distinct retriever IDs in the resulting
    :class:`~arandu.shared.rag.schemas.RetrievalRecord`.
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
            chunker_id: Chunker view ID this index was built from. Used to
                derive ``retriever_id`` and not validated against the manifest.
            language: ``"pt"`` or ``"en"``. Selects the tokenizer applied to
                queries; should match the language used at index-build time.

        Raises:
            FileNotFoundError: If ``index_dir`` lacks either index file.
            ValueError: If ``language`` is unsupported.
        """
        self.retriever_id = f"bm25_{chunker_id}"
        self._chunker_id = chunker_id
        self._language = language
        self._tokenize = _make_tokenizer(language)
        self._bm25, self._chunk_ids = self._load_index(index_dir)

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
        with (index_dir / INDEX_FILENAME).open("wb") as f:
            pickle.dump(bm25, f)
        manifest = {
            "chunker_id": chunker_id,
            "language": language,
            "chunks_indexed": len(chunks),
            "built_at": datetime.now(UTC).isoformat(),
            "chunk_ids": chunk_ids,
        }
        (index_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def _load_index(index_dir: Path) -> tuple[BM25Okapi, list[str]]:
        idx_path = index_dir / INDEX_FILENAME
        mfst_path = index_dir / MANIFEST_FILENAME
        if not idx_path.exists() or not mfst_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_dir} "
                f"(expected '{INDEX_FILENAME}' and '{MANIFEST_FILENAME}')."
            )
        with idx_path.open("rb") as f:
            bm25 = pickle.load(f)
        manifest = json.loads(mfst_path.read_text())
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
