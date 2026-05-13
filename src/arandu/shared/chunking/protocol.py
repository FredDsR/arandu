"""Chunker Protocol — narrow contract for chunker implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from arandu.shared.chunking.schemas import Chunk


@runtime_checkable
class Chunker(Protocol):
    """A chunker slices text into ``Chunk`` references over char offsets.

    Implementations must expose a stable ``chunker_id`` and a ``chunk`` method
    that returns offsets-only ``Chunk`` objects (no text payload).
    """

    chunker_id: str

    def chunk(self, text: str, source_file_id: str) -> list[Chunk]:
        """Slice ``text`` and return offsets-only chunks tagged with ``source_file_id``."""
        ...
