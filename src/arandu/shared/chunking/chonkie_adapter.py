"""Adapter wrapping a chonkie chunker as a Chunker."""

from __future__ import annotations

import hashlib
from typing import Any

from arandu.shared.chunking.schemas import Chunk


def chunk_id_for(source_file_id: str, chunker_id: str, start_char: int, end_char: int) -> str:
    """Compute the stable 16-char sha1 prefix used as ``chunk_id``."""
    payload = f"{source_file_id}|{chunker_id}|{start_char}|{end_char}".encode()
    return hashlib.sha1(payload).hexdigest()[:16]


class ChonkieAdapter:
    """Adapter that wraps a chonkie chunker, emitting offsets-only ``Chunk`` objects.

    The wrapped chonkie object must yield items exposing ``start_index``, ``end_index``,
    and ``token_count`` (this is the contract for ``RecursiveChunker``, ``TokenChunker``,
    and ``SentenceChunker`` in chonkie 1.x).
    """

    chunker_id: str

    def __init__(
        self,
        chunker_id: str,
        chonkie_chunker: Any,
        tokenizer_id: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            chunker_id: Identifier of this chunker view (e.g. ``cep_4k``).
            chonkie_chunker: A chonkie chunker instance with a ``chunk(text)`` method.
            tokenizer_id: Optional tokenizer identifier recorded on every emitted ``Chunk``.
        """
        self.chunker_id = chunker_id
        self._impl = chonkie_chunker
        self._tokenizer_id = tokenizer_id

    def chunk(self, text: str, source_file_id: str) -> list[Chunk]:
        """Slice ``text`` via the wrapped chonkie chunker and emit offsets-only chunks."""
        if not text:
            return []

        results: list[Chunk] = []
        for item in self._impl.chunk(text):
            start = int(item.start_index)
            end = int(item.end_index)
            if end <= start:
                continue
            token_count = getattr(item, "token_count", None)
            results.append(
                Chunk(
                    chunk_id=chunk_id_for(source_file_id, self.chunker_id, start, end),
                    source_file_id=source_file_id,
                    chunker_id=self.chunker_id,
                    start_char=start,
                    end_char=end,
                    token_count=int(token_count) if token_count is not None else None,
                    tokenizer_id=self._tokenizer_id,
                )
            )
        return results
