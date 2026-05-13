"""Pydantic schemas for the shared chunking module.

Defines an offsets-only ``Chunk`` reference and a ``ChunkSet`` container that
holds multiple chunker views over a single source transcription. Chunks do not
carry text — text is resolved on demand from the source ``EnrichedRecord`` by
``ChunkResolver`` using the recorded char span.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — Pydantic field type, needs runtime access
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationInfo, field_validator

if TYPE_CHECKING:
    from typing import Self


class Chunk(BaseModel):
    """Offsets-only chunk reference into a source transcription.

    The chunk does not carry text. Use ``ChunkResolver.text(chunk)`` to materialize
    the text on demand from the source ``EnrichedRecord``.

    Attributes:
        chunk_id: Stable content hash, sha1(``source_file_id|chunker_id|start|end``)[:16].
        source_file_id: ID of the source transcription (matches ``EnrichedRecord.file_id``).
        chunker_id: Identifier of the chunker view (e.g. ``cep_4k``, ``bm25_512t``).
        start_char: Inclusive start offset (char) in the source text.
        end_char: Exclusive end offset (char) in the source text. Must be > start_char.
        token_count: Advisory token count; meaningful only when ``tokenizer_id`` is set.
        tokenizer_id: Tokenizer used to derive ``token_count``.
    """

    chunk_id: str = Field(
        ..., min_length=1, description="Stable content hash (16-char sha1 prefix)"
    )
    source_file_id: str = Field(..., min_length=1)
    chunker_id: str = Field(..., min_length=1)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., gt=0)
    token_count: int | None = Field(default=None, ge=0)
    tokenizer_id: str | None = Field(default=None)

    @field_validator("end_char")
    @classmethod
    def _end_after_start(cls, v: int, info: ValidationInfo) -> int:
        """Ensure end_char > start_char."""
        start = info.data.get("start_char")
        if start is not None and v <= start:
            raise ValueError("end_char must be > start_char")
        return v


class ChunkSet(BaseModel):
    """All chunker views over a single source transcription.

    Each view is keyed by ``chunker_id`` and holds an ordered list of ``Chunk``
    objects covering the source. The ``source_text_sha256`` snapshot enables
    drift detection at resolve time.
    """

    source_file_id: str = Field(..., min_length=1)
    source_filename: str = Field(..., min_length=1)
    source_text_sha256: str = Field(
        ..., min_length=64, max_length=64, description="SHA-256 of source text at chunking time"
    )
    views: dict[str, list[Chunk]] = Field(
        ..., description="Map of chunker_id to ordered chunks for that view."
    )
    generated_at: datetime

    def view(self, chunker_id: str) -> list[Chunk]:
        """Return chunks for ``chunker_id``.

        Raises:
            KeyError: If the view is not present; the error lists available view IDs.
        """
        if chunker_id not in self.views:
            available = sorted(self.views)
            raise KeyError(f"chunker_id '{chunker_id}' not in this ChunkSet (have: {available})")
        return self.views[chunker_id]

    def lookup(self, chunk_id: str) -> Chunk:
        """Find a chunk by ID across all views."""
        for chunks in self.views.values():
            for chunk in chunks:
                if chunk.chunk_id == chunk_id:
                    return chunk
        raise KeyError(f"chunk_id '{chunk_id}' not found in ChunkSet")

    def overlapping(self, start: int, end: int, chunker_id: str) -> list[Chunk]:
        """Return chunks in ``chunker_id`` overlapping the half-open interval [start, end).

        Raises:
            KeyError: If the view is not present.
        """
        return [
            c for c in self.view(chunker_id) if not (c.end_char <= start or c.start_char >= end)
        ]

    def save(self, path: str | Path) -> None:
        """Serialize this ChunkSet to ``path`` as JSON."""
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a ChunkSet from ``path``."""
        return cls.model_validate_json(Path(path).read_text())
