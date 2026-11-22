"""ChunkResolver — resolves Chunk references to source text on demand."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from arandu.shared.chunking.schemas import Chunk


class SourceDriftError(RuntimeError):
    """Raised when the loaded source text's SHA-256 does not match what was recorded.

    Indicates the underlying transcription changed between chunking and resolve
    time — every downstream chunk_id for that source becomes invalid.
    """


class ChunkResolver:
    """Resolve ``Chunk`` references to substrings of the source transcription.

    The resolver is constructed with a ``text_loader`` callable that returns the
    full source text for a given ``source_file_id``. Loaded texts are cached in
    an LRU. When ``expected_sha256_by_file_id`` is provided, every load is
    verified against the recorded hash; mismatches raise :class:`SourceDriftError`.
    """

    def __init__(
        self,
        text_loader: Callable[[str], str],
        expected_sha256_by_file_id: dict[str, str] | None = None,
        cache_size: int = 64,
    ) -> None:
        """Initialize the resolver.

        Args:
            text_loader: Callable mapping ``source_file_id`` → full source text.
            expected_sha256_by_file_id: Optional mapping from ``source_file_id``
                to its expected SHA-256 hash. When a source_file_id appears in
                this mapping, the loaded text is hashed and compared on first
                load; mismatch raises :class:`SourceDriftError`.
            cache_size: LRU cache size for loaded source texts.
        """
        self._loader = text_loader
        self._expected = expected_sha256_by_file_id or {}
        self._load_cached = lru_cache(maxsize=cache_size)(self._load_and_verify)

    def text(self, chunk: Chunk) -> str:
        """Return the substring ``source_text[chunk.start_char:chunk.end_char]``.

        Raises:
            SourceDriftError: If the loaded source text's SHA-256 doesn't match
                the expected hash recorded at chunking time.
        """
        full = self._load_cached(chunk.source_file_id)
        return full[chunk.start_char : chunk.end_char]

    def texts(self, chunks: list[Chunk]) -> list[str]:
        """Resolve a batch of chunks in order."""
        return [self.text(c) for c in chunks]

    def _load_and_verify(self, source_file_id: str) -> str:
        """Load source text and verify SHA-256 if expected hash is recorded."""
        text = self._loader(source_file_id)
        expected = self._expected.get(source_file_id)
        if expected is not None:
            observed = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if observed != expected:
                raise SourceDriftError(
                    f"Source text for '{source_file_id}' has drifted: "
                    f"expected sha256={expected[:8]}…, observed sha256={observed[:8]}…"
                )
        return text
