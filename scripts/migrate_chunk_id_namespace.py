#!/usr/bin/env python3
"""Migrate a run's ``chunk_id`` namespace onto the canonical coordinate space.

Before ``EnrichedRecord.transcription_text`` was normalized, the ``chunk`` stage
chunked the raw transcription while CEP generation chunked the same text
``.strip()``ed. Whisper prefixes its output with a space, so every stage
boundary sat one character past generation's and the two produced disjoint
``chunk_id`` namespaces: 0 of 2670 pairs in ``thesis-run-01`` resolved against
their persisted ``ChunkSet``.

This rewrites a run's ``chunk`` stage in the canonical space, by re-chunking the
canonical text with the real chunker rather than shifting offsets
arithmetically, and remaps every consumer that referenced the old ids. Design:
``docs/superpowers/specs/2026-09-09-chunk-id-coordinate-space-normalization-design.md``.

Usage:
    uv run python scripts/migrate_chunk_id_namespace.py --id thesis-run-02 --dry-run
    uv run python scripts/migrate_chunk_id_namespace.py --id thesis-run-02
    uv run python scripts/migrate_chunk_id_namespace.py --id thesis-run-02 --verify
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - runtime use in signatures

from arandu.shared.chunking.registry import get_chunker
from arandu.shared.chunking.schemas import Chunk, ChunkSet
from arandu.shared.io import resolve_transcription_path
from arandu.shared.schemas import EnrichedRecord


@dataclass(frozen=True)
class SourceTexts:
    """The two readings of one transcription file the migration needs.

    After the ``EnrichedRecord`` validator landed, the raw text is no longer
    observable through the schema: loading a record already returns it stripped.
    The migration still needs the raw lead to shift the atlas passage offsets
    and to assert the rewrite is a pure shift, so it reads the file both ways.

    Attributes:
        canonical: Text as every offset in the pipeline refers to it.
        lead_ws: Count of leading whitespace characters the schema stripped.
        filename: The record's source media filename, available to callers.
            ``rewrite_chunk_sets`` does not use it: the rewritten ChunkSet keeps
            the old ChunkSet's own ``source_filename``, per the migration's
            invariant that filenames are untouched.
    """

    canonical: str
    lead_ws: int
    filename: str


def load_source_texts(transcription_dir: Path, file_id: str) -> SourceTexts:
    """Read one transcription both canonically and raw.

    Args:
        transcription_dir: The run's ``transcription/outputs`` directory.
        file_id: Source file identifier.

    Returns:
        Both readings, plus the stripped leading-whitespace count.

    Raises:
        FileNotFoundError: If no transcription file exists for ``file_id``. This
            aborts rather than skipping: a ChunkSet with no transcription is an
            inconsistency the migration must not paper over.
    """
    path = resolve_transcription_path(transcription_dir, file_id)
    if path is None:
        raise FileNotFoundError(f"No transcription for {file_id!r} under {transcription_dir}")

    payload = path.read_text(encoding="utf-8")
    raw = json.loads(payload)["transcription_text"]
    record = EnrichedRecord.model_validate_json(payload)
    return SourceTexts(
        canonical=record.transcription_text,
        lead_ws=len(raw) - len(raw.lstrip()),
        filename=record.name,
    )


def _assert_pure_shift(path: Path, stale: list[Chunk], fresh: list[Chunk], lead_ws: int) -> None:
    """Abort unless the fresh chunking is the stale one shifted by ``lead_ws``.

    This is the safety rail on re-chunking rather than shifting arithmetically,
    and it doubles as the content-preservation proof: identical spans over the
    same underlying text resolve to identical strings, the sole exception being
    the first chunk, which loses the stripped leading whitespace.

    Args:
        path: ChunkSet path, for the error message.
        stale: Chunks currently on disk.
        fresh: Chunks produced from the canonical text.
        lead_ws: Leading whitespace the schema stripped.

    Raises:
        ValueError: If the count or any span fails to match the expected shift.
    """
    if len(stale) != len(fresh):
        raise ValueError(
            f"{path}: chunk count changed under normalization "
            f"({len(stale)} -> {len(fresh)}); refusing to write"
        )
    for old, new in zip(stale, fresh, strict=True):
        expected = (max(old.start_char - lead_ws, 0), old.end_char - lead_ws)
        if (new.start_char, new.end_char) != expected:
            raise ValueError(
                f"{path}: expected chunk span {expected}, got "
                f"({new.start_char}, {new.end_char}); refusing to write"
            )


def rewrite_chunk_sets(run_dir: Path, *, dry_run: bool) -> tuple[dict[str, str], dict[str, int]]:
    """Rewrite every ChunkSet in the run's canonical coordinate space.

    Args:
        run_dir: The run directory (``<results base>/<run id>``).
        dry_run: When true, compute everything but write nothing.

    Returns:
        A tuple of the ``old chunk_id -> new chunk_id`` map and the
        ``file_id -> stripped leading whitespace`` map.

    Raises:
        ValueError: If a ChunkSet file holds views other than the one its
            directory names, or if the rewrite is not a pure shift.
    """
    chunk_outputs = run_dir / "chunk" / "outputs"
    transcription_dir = run_dir / "transcription" / "outputs"
    id_map: dict[str, str] = {}
    lead_by_file: dict[str, int] = {}

    for view_dir in sorted(p for p in chunk_outputs.iterdir() if p.is_dir()):
        view_id = view_dir.name
        chunker = get_chunker(view_id)
        for path in sorted(view_dir.glob("*.json")):
            stale_set = ChunkSet.load(path)
            if set(stale_set.views) != {view_id}:
                raise ValueError(
                    f"{path}: expected only the {view_id!r} view, found {sorted(stale_set.views)}"
                )

            texts = load_source_texts(transcription_dir, stale_set.source_file_id)
            lead_by_file[stale_set.source_file_id] = texts.lead_ws

            stale = stale_set.view(view_id)
            fresh = chunker.chunk(texts.canonical, source_file_id=stale_set.source_file_id)
            _assert_pure_shift(path, stale, fresh, texts.lead_ws)

            for old, new in zip(stale, fresh, strict=True):
                id_map[old.chunk_id] = new.chunk_id

            if not dry_run:
                ChunkSet(
                    source_file_id=stale_set.source_file_id,
                    source_filename=stale_set.source_filename,
                    source_text_sha256=hashlib.sha256(texts.canonical.encode("utf-8")).hexdigest(),
                    views={view_id: fresh},
                    generated_at=stale_set.generated_at,
                ).save(path)

    return id_map, lead_by_file
