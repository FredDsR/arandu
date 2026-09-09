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
    remap_bm25_manifests,
    remap_passage_chunk_ids,
    rewrite_chunk_sets,
    shift_passage_offsets,
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


def write_retrieval_output(
    run_dir: Path, stage: str, arm: str, name: str, chunk_ids: list[str]
) -> Path:
    """Write a minimal retrieval-shaped artifact carrying ``chunk_ids``."""
    directory = run_dir / stage / "outputs" / arm / "cep"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "qa_pair_id": f"file-1:{name}:0",
                "question": "Pergunta?",
                "retriever_id": f"{arm}_cep_4k",
                "chunker_id": VIEW,
                "top_k": len(chunk_ids),
                "passages": [
                    {"chunk_id": cid, "rank": i, "score": 1.0, "payload": None}
                    for i, cid in enumerate(chunk_ids)
                ],
            }
        )
    )
    return path


class TestRemapBm25Manifests:
    """The BM25 index manifest lists chunk_ids positionally against bm25.pkl."""

    def test_remaps_known_ids_and_leaves_others_alone(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        directory = run / "retrieve" / "indexes" / "bm25_cep_4k"
        directory.mkdir(parents=True)
        manifest = directory / "manifest.json"
        manifest.write_text(
            json.dumps({"sha256": "deadbeef", "chunk_ids": ["old-a", "unknown", "old-b"]})
        )

        remapped = remap_bm25_manifests(run, {"old-a": "new-a", "old-b": "new-b"}, dry_run=False)

        assert remapped == 2
        written = json.loads(manifest.read_text())
        assert written["chunk_ids"] == ["new-a", "unknown", "new-b"]
        # The manifest's sha256 covers bm25.pkl, not the manifest, so it stays.
        assert written["sha256"] == "deadbeef"

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        directory = run / "retrieve" / "indexes" / "bm25_cep_4k"
        directory.mkdir(parents=True)
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({"chunk_ids": ["old-a"]}))
        before = manifest.read_text()

        assert remap_bm25_manifests(run, {"old-a": "new-a"}, dry_run=True) == 1
        assert manifest.read_text() == before


class TestRemapPassageChunkIds:
    """Only offset-derived ids are in the map, so other namespaces survive."""

    def test_remaps_across_all_three_stages(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        paths = [
            write_retrieval_output(run, stage, "bm25", "q0", ["old-a", "old-b"])
            for stage in ("retrieve", "answers", "judge_answers")
        ]

        files, refs = remap_passage_chunk_ids(
            run, {"old-a": "new-a", "old-b": "new-b"}, dry_run=False
        )

        assert (files, refs) == (3, 6)
        for path in paths:
            ids = [p["chunk_id"] for p in json.loads(path.read_text())["passages"]]
            assert ids == ["new-a", "new-b"]

    def test_leaves_foreign_namespaces_untouched(self, tmp_path: Path) -> None:
        """atlas_rag/khop_passage use <file_id>:<index>, khop_triple uses triple:<sha>."""
        run = tmp_path / "run"
        atlas = write_retrieval_output(run, "retrieve", "atlas_rag", "q0", ["file-9:3"])
        triple = write_retrieval_output(run, "retrieve", "khop_triple", "q0", ["triple:abc123"])

        files, refs = remap_passage_chunk_ids(run, {"old-a": "new-a"}, dry_run=False)

        assert (files, refs) == (0, 0)
        assert json.loads(atlas.read_text())["passages"][0]["chunk_id"] == "file-9:3"
        assert json.loads(triple.read_text())["passages"][0]["chunk_id"] == "triple:abc123"

    def test_skips_artifacts_without_passages(self, tmp_path: Path) -> None:
        """The null arm carries no passages; run_metadata.json carries no ids."""
        run = tmp_path / "run"
        directory = run / "retrieve" / "outputs" / "null" / "cep"
        directory.mkdir(parents=True)
        (directory / "q0.json").write_text(json.dumps({"qa_pair_id": "file-1:x:0"}))

        assert remap_passage_chunk_ids(run, {"old-a": "new-a"}, dry_run=False) == (0, 0)

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        path = write_retrieval_output(run, "retrieve", "bm25", "q0", ["old-a"])
        before = path.read_text()

        assert remap_passage_chunk_ids(run, {"old-a": "new-a"}, dry_run=True) == (1, 1)
        assert path.read_text() == before


class TestShiftPassageOffsets:
    """Atlas passage offsets live in EnrichedRecord space, which just moved."""

    def _write_sidecar(self, run_dir: Path, offsets: list[dict[str, Any]]) -> Path:
        directory = run_dir / "kg" / "outputs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "passage_offsets.json"
        path.write_text(
            json.dumps(
                {
                    "kg_run_id": "run",
                    "offsets": offsets,
                    "unmatched": [],
                    "generated_at": "2026-06-24T18:40:49.678102Z",
                }
            )
        )
        return path

    def test_shifts_by_the_files_stripped_lead(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        path = self._write_sidecar(
            run,
            [
                {
                    "passage_id": "file-1:0",
                    "source_file_id": "file-1",
                    "start_char": 10,
                    "end_char": 100,
                    "chunker_id": "atlas_8k",
                }
            ],
        )

        assert shift_passage_offsets(run, {"file-1": 1}, dry_run=False) == 1

        offset = json.loads(path.read_text())["offsets"][0]
        assert (offset["start_char"], offset["end_char"]) == (9, 99)

    def test_clamps_a_zero_start_at_zero(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        path = self._write_sidecar(
            run,
            [
                {
                    "passage_id": "file-1:0",
                    "source_file_id": "file-1",
                    "start_char": 0,
                    "end_char": 50,
                    "chunker_id": "atlas_8k",
                }
            ],
        )

        shift_passage_offsets(run, {"file-1": 1}, dry_run=False)

        offset = json.loads(path.read_text())["offsets"][0]
        assert (offset["start_char"], offset["end_char"]) == (0, 49)

    def test_leaves_files_with_no_stripped_lead_alone(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        path = self._write_sidecar(
            run,
            [
                {
                    "passage_id": "file-2:0",
                    "source_file_id": "file-2",
                    "start_char": 10,
                    "end_char": 100,
                    "chunker_id": "atlas_8k",
                }
            ],
        )
        before = path.read_text()

        assert shift_passage_offsets(run, {"file-2": 0}, dry_run=False) == 0
        assert path.read_text() == before

    def test_returns_zero_when_the_sidecar_is_absent(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        run.mkdir()

        assert shift_passage_offsets(run, {"file-1": 1}, dry_run=False) == 0
