"""Tests for the chunk_id namespace migration script."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from arandu.shared.chunking.registry import get_chunker
from arandu.shared.chunking.schemas import ChunkSet
from scripts.migrate_chunk_id_namespace import (
    load_source_texts,
    rewrite_chunk_sets,
)

if TYPE_CHECKING:
    from pathlib import Path

    from arandu.shared.chunking.schemas import Chunk

VIEW = "cep_4k"

# Long enough that cep_4k emits several chunks. Stripped so BODY itself is
# already canonical: the repeated unit ends in ". " and the multiplication
# would otherwise leave a trailing space that the EnrichedRecord validator
# (a full .strip()) would remove along with the leading space the tests add,
# breaking the "lead-only shift" assumption the assertions below rely on.
BODY = (
    (
        "O pescador contou que quando o rio sobe ele guarda o barco no barranco alto. "
        "Depois falou da prefeitura, do ciclone e da ajuda que veio da universidade. "
    )
    * 120
).strip()


def write_transcription(run_dir: Path, file_id: str, raw_text: str) -> None:
    """Write a transcription whose stored text is ``raw_text`` verbatim."""
    directory = run_dir / "transcription" / "outputs"
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "file_id": file_id,
        "name": f"{file_id}.mp3",
        "mimeType": "audio/mpeg",
        "parents": ["folder"],
        "webContentLink": "https://drive.google.com/test",
        "size_bytes": 1024,
        "duration_milliseconds": 60000,
        "transcription_text": raw_text,
        "detected_language": "pt",
        "language_probability": 0.95,
        "model_id": "whisper-large-v3",
        "compute_device": "cpu",
        "processing_duration_sec": 10.0,
        "transcription_status": "completed",
    }
    (directory / f"{file_id}_transcription.json").write_text(json.dumps(payload))


def write_stale_chunk_set(run_dir: Path, file_id: str, raw_text: str) -> list[Chunk]:
    """Write the ChunkSet the buggy chunk stage would have produced.

    The buggy stage chunked the raw text, so its offsets sit ``lead_ws`` past
    the canonical ones. Returns the stale chunks for the caller to assert on.
    """
    directory = run_dir / "chunk" / "outputs" / VIEW
    directory.mkdir(parents=True, exist_ok=True)
    chunks = get_chunker(VIEW).chunk(raw_text, source_file_id=file_id)
    ChunkSet(
        source_file_id=file_id,
        source_filename=f"{file_id}.mp3",
        source_text_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        views={VIEW: chunks},
        generated_at=datetime(2026, 6, 24, tzinfo=UTC),
    ).save(directory / f"{file_id}.json")
    return chunks


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A synthetic run carrying one file with Whisper's leading space."""
    run = tmp_path / "thesis-run-02"
    write_transcription(run, "file-1", f" {BODY}")
    write_stale_chunk_set(run, "file-1", f" {BODY}")
    return run


class TestLoadSourceTexts:
    """load_source_texts must see both readings of the transcription."""

    def test_returns_canonical_text_and_the_stripped_lead(self, run_dir: Path) -> None:
        texts = load_source_texts(run_dir / "transcription" / "outputs", "file-1")

        assert texts.canonical == BODY
        assert texts.lead_ws == 1
        assert texts.filename == "file-1.mp3"

    def test_lead_is_zero_when_the_raw_text_is_already_canonical(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        write_transcription(run, "file-2", BODY)

        texts = load_source_texts(run / "transcription" / "outputs", "file-2")

        assert texts.lead_ws == 0

    def test_raises_when_the_transcription_is_missing(self, tmp_path: Path) -> None:
        directory = tmp_path / "outputs"
        directory.mkdir()

        with pytest.raises(FileNotFoundError, match="ghost"):
            load_source_texts(directory, "ghost")


class TestRewriteChunkSets:
    """rewrite_chunk_sets re-chunks the canonical text with the real chunker."""

    def test_maps_every_stale_id_to_its_canonical_id(self, run_dir: Path) -> None:
        stale = ChunkSet.load(run_dir / "chunk" / "outputs" / VIEW / "file-1.json")
        stale_ids = [c.chunk_id for c in stale.view(VIEW)]

        id_map, lead_by_file = rewrite_chunk_sets(run_dir, dry_run=False)

        assert lead_by_file == {"file-1": 1}
        assert set(id_map) == set(stale_ids)
        assert all(old != new for old, new in id_map.items())

    def test_written_chunk_set_matches_a_fresh_canonical_chunking(self, run_dir: Path) -> None:
        expected = get_chunker(VIEW).chunk(BODY, source_file_id="file-1")

        rewrite_chunk_sets(run_dir, dry_run=False)

        written = ChunkSet.load(run_dir / "chunk" / "outputs" / VIEW / "file-1.json")
        assert [c.chunk_id for c in written.view(VIEW)] == [c.chunk_id for c in expected]
        assert written.source_text_sha256 == hashlib.sha256(BODY.encode("utf-8")).hexdigest()

    def test_offsets_shift_back_by_the_stripped_lead(self, run_dir: Path) -> None:
        stale = ChunkSet.load(run_dir / "chunk" / "outputs" / VIEW / "file-1.json")
        stale_spans = [(c.start_char, c.end_char) for c in stale.view(VIEW)]

        rewrite_chunk_sets(run_dir, dry_run=False)

        written = ChunkSet.load(run_dir / "chunk" / "outputs" / VIEW / "file-1.json")
        new_spans = [(c.start_char, c.end_char) for c in written.view(VIEW)]
        assert new_spans == [(max(start - 1, 0), end - 1) for start, end in stale_spans]

    def test_dry_run_builds_the_map_without_touching_disk(self, run_dir: Path) -> None:
        path = run_dir / "chunk" / "outputs" / VIEW / "file-1.json"
        before = path.read_text()

        id_map, _ = rewrite_chunk_sets(run_dir, dry_run=True)

        assert id_map
        assert path.read_text() == before
