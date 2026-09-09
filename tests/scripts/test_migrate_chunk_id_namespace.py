"""Tests for the chunk_id namespace migration script."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from arandu.shared.chunking.registry import get_chunker
from arandu.shared.chunking.schemas import Chunk, ChunkSet
from scripts.migrate_chunk_id_namespace import (
    _assert_pure_shift,
    load_source_texts,
    main,
    remap_bm25_manifests,
    remap_passage_chunk_ids,
    rewrite_chunk_sets,
    shift_passage_offsets,
    verify,
)

if TYPE_CHECKING:
    from pathlib import Path

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


def make_chunk(file_id: str, start: int, end: int) -> Chunk:
    """Build a Chunk with an arbitrary but well-formed id."""
    return Chunk(
        chunk_id=f"{start:08x}{end:08x}",
        source_file_id=file_id,
        chunker_id=VIEW,
        start_char=start,
        end_char=end,
    )


def write_pipeline_metadata(run_dir: Path, *, replicated: bool) -> Path:
    """Write the run's ``pipeline.json``, optionally with clone provenance.

    ``arandu replicate`` stamps ``PipelineMetadata.replicated_from``; the
    migration refuses to write to a run that carries no such provenance.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "pipeline.json"
    payload: dict[str, Any] = {
        "pipeline_id": run_dir.name,
        "created_at": "2026-06-24T18:36:30.403511Z",
        "steps_run": ["transcription", "chunk"],
        "schema_version": "2.0",
        "replicated_from": (
            {
                "source_pipeline_id": "thesis-run-01",
                "replicated_at": "2026-06-24T18:36:30.403511Z",
            }
            if replicated
            else None
        ),
    }
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A synthetic run carrying one file with Whisper's leading space."""
    run = tmp_path / "thesis-run-02"
    write_transcription(run, "file-1", f" {BODY}")
    write_stale_chunk_set(run, "file-1", f" {BODY}")
    write_pipeline_metadata(run, replicated=True)
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

    def test_writes_nothing_when_the_canonical_chunking_diverges(self, run_dir: Path) -> None:
        """The safety rail must abort before touching the ChunkSet on disk."""
        path = run_dir / "chunk" / "outputs" / VIEW / "file-1.json"
        ChunkSet(
            source_file_id="file-1",
            source_filename="file-1.mp3",
            source_text_sha256=hashlib.sha256(f" {BODY}".encode()).hexdigest(),
            views={VIEW: [make_chunk("file-1", 1, 400)]},
            generated_at=datetime(2026, 6, 24, tzinfo=UTC),
        ).save(path)
        before = path.read_text()

        with pytest.raises(ValueError, match="chunk count changed"):
            rewrite_chunk_sets(run_dir, dry_run=False)

        assert path.read_text() == before

    def test_rejects_a_chunk_set_holding_more_than_its_directorys_view(self, run_dir: Path) -> None:
        """A multi-view file under ``<view>/`` would lose its other views."""
        path = run_dir / "chunk" / "outputs" / VIEW / "file-1.json"
        stale = ChunkSet.load(path)
        ChunkSet(
            source_file_id=stale.source_file_id,
            source_filename=stale.source_filename,
            source_text_sha256=stale.source_text_sha256,
            views={VIEW: stale.view(VIEW), "bm25_512t": [make_chunk("file-1", 1, 400)]},
            generated_at=stale.generated_at,
        ).save(path)
        before = path.read_text()

        with pytest.raises(ValueError, match="expected only the 'cep_4k' view"):
            rewrite_chunk_sets(run_dir, dry_run=False)

        assert path.read_text() == before

    def test_reports_a_chunk_set_built_from_another_text(self, run_dir: Path) -> None:
        """Neither hash matching means the ChunkSet is not this file's."""
        path = run_dir / "chunk" / "outputs" / VIEW / "file-1.json"
        stale = ChunkSet.load(path)
        ChunkSet(
            source_file_id=stale.source_file_id,
            source_filename=stale.source_filename,
            source_text_sha256=hashlib.sha256(b"some other transcription").hexdigest(),
            views={VIEW: stale.view(VIEW)},
            generated_at=stale.generated_at,
        ).save(path)
        before = path.read_text()

        with pytest.raises(ValueError, match="was not built from"):
            rewrite_chunk_sets(run_dir, dry_run=False)

        assert path.read_text() == before

    def test_is_idempotent_on_an_already_migrated_run(self, run_dir: Path) -> None:
        """A second pass recognizes the canonical hash and leaves the run alone."""
        rewrite_chunk_sets(run_dir, dry_run=False)
        path = run_dir / "chunk" / "outputs" / VIEW / "file-1.json"
        before = path.read_text()

        id_map, lead_by_file = rewrite_chunk_sets(run_dir, dry_run=False)

        assert id_map == {}
        assert lead_by_file == {"file-1": 0}
        assert path.read_text() == before


class TestAssertPureShift:
    """The safety rail behind re-chunking instead of shifting arithmetically."""

    def test_accepts_the_expected_shift(self, tmp_path: Path) -> None:
        stale = [make_chunk("file-1", 1, 400), make_chunk("file-1", 400, 900)]
        fresh = [make_chunk("file-1", 0, 399), make_chunk("file-1", 399, 899)]

        _assert_pure_shift(tmp_path / "file-1.json", stale, fresh, 1)

    def test_raises_when_the_chunk_count_changed(self, tmp_path: Path) -> None:
        stale = [make_chunk("file-1", 1, 400), make_chunk("file-1", 400, 900)]
        fresh = [make_chunk("file-1", 0, 899)]

        with pytest.raises(ValueError, match="chunk count changed"):
            _assert_pure_shift(tmp_path / "file-1.json", stale, fresh, 1)

    def test_raises_when_a_span_does_not_match_the_shift(self, tmp_path: Path) -> None:
        stale = [make_chunk("file-1", 1, 400)]
        fresh = [make_chunk("file-1", 1, 400)]

        with pytest.raises(ValueError, match="refusing to write"):
            _assert_pure_shift(tmp_path / "file-1.json", stale, fresh, 1)

    def test_clamps_the_first_span_at_zero(self, tmp_path: Path) -> None:
        """The first chunk loses the stripped lead instead of going negative."""
        stale = [make_chunk("file-1", 0, 400)]
        fresh = [make_chunk("file-1", 0, 399)]

        _assert_pure_shift(tmp_path / "file-1.json", stale, fresh, 1)


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

    def _offset(self, file_id: str, start: int, end: int) -> dict[str, Any]:
        """One sidecar entry for ``file_id`` spanning ``[start, end)``."""
        return {
            "passage_id": f"{file_id}:0",
            "source_file_id": file_id,
            "start_char": start,
            "end_char": end,
            "chunker_id": "atlas_8k",
        }

    def test_shifts_by_the_files_stripped_lead(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        write_transcription(run, "file-1", f" {BODY}")
        path = self._write_sidecar(run, [self._offset("file-1", 10, 100)])

        assert shift_passage_offsets(run, {"file-1": 1}, dry_run=False) == 1

        offset = json.loads(path.read_text())["offsets"][0]
        assert (offset["start_char"], offset["end_char"]) == (9, 99)

    def test_clamps_a_zero_start_at_zero(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        write_transcription(run, "file-1", f" {BODY}")
        path = self._write_sidecar(run, [self._offset("file-1", 0, 50)])

        shift_passage_offsets(run, {"file-1": 1}, dry_run=False)

        offset = json.loads(path.read_text())["offsets"][0]
        assert (offset["start_char"], offset["end_char"]) == (0, 49)

    def test_leaves_files_with_no_stripped_lead_alone(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        path = self._write_sidecar(run, [self._offset("file-2", 10, 100)])
        before = path.read_text()

        assert shift_passage_offsets(run, {"file-2": 0}, dry_run=False) == 0
        assert path.read_text() == before

    def test_returns_zero_when_the_sidecar_is_absent(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        run.mkdir()

        assert shift_passage_offsets(run, {"file-1": 1}, dry_run=False) == 0

    def test_aborts_when_the_file_has_no_chunk_set(self, tmp_path: Path) -> None:
        """A missing key is not a lead of zero: the stages may select differently."""
        run = tmp_path / "run"
        write_transcription(run, "file-1", f" {BODY}")
        path = self._write_sidecar(run, [self._offset("file-1", 10, 100)])
        before = path.read_text()

        with pytest.raises(ValueError, match="file-1"):
            shift_passage_offsets(run, {}, dry_run=False)

        assert path.read_text() == before

    def test_aborts_when_the_shifted_span_resolves_to_other_text(self, tmp_path: Path) -> None:
        """The sidecar's old state is in hand, so the shift is checked before writing."""
        run = tmp_path / "run"
        write_transcription(run, "file-1", f" {BODY}")
        path = self._write_sidecar(run, [self._offset("file-1", 10, 100)])
        before = path.read_text()

        with pytest.raises(ValueError, match="resolves to different text"):
            shift_passage_offsets(run, {"file-1": 5}, dry_run=False)

        assert path.read_text() == before

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        run = tmp_path / "run"
        write_transcription(run, "file-1", f" {BODY}")
        path = self._write_sidecar(run, [self._offset("file-1", 10, 100)])
        before = path.read_text()

        assert shift_passage_offsets(run, {"file-1": 1}, dry_run=True) == 1
        assert path.read_text() == before


def write_cep_record(run_dir: Path, file_id: str, chunk_ids: list[str]) -> Path:
    """Write a CEP QA record whose pairs point at ``chunk_ids``."""
    directory = run_dir / "cep" / "outputs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{file_id}_cep_qa.json"
    path.write_text(
        json.dumps(
            {
                "source_gdrive_id": file_id,
                "source_filename": f"{file_id}.mp3",
                "transcription_text": BODY,
                "model_id": "qwen3:14b",
                "provider": "ollama",
                "language": "pt",
                "total_pairs": len(chunk_ids),
                "qa_pairs": [
                    {
                        "question": f"Pergunta {i}?",
                        "answer": "Resposta.",
                        "context": "Trecho.",
                        "question_type": "factual",
                        "bloom_level": "remember",
                        "chunk_id": cid,
                    }
                    for i, cid in enumerate(chunk_ids)
                ],
            }
        )
    )
    return path


class TestVerify:
    """verify() is the migration's proof, run after the rewrite."""

    def test_passes_on_a_fully_migrated_run(self, run_dir: Path) -> None:
        fresh_ids = [c.chunk_id for c in get_chunker(VIEW).chunk(BODY, source_file_id="file-1")]
        write_cep_record(run_dir, "file-1", fresh_ids)
        id_map, lead_by_file = rewrite_chunk_sets(run_dir, dry_run=False)
        shift_passage_offsets(run_dir, lead_by_file, dry_run=False)

        assert verify(run_dir).failures == []
        assert len(id_map) == len(fresh_ids)

    def test_reports_cep_pairs_that_do_not_resolve(self, run_dir: Path) -> None:
        write_cep_record(run_dir, "file-1", ["not-a-real-chunk-id"])
        rewrite_chunk_sets(run_dir, dry_run=False)

        failures = verify(run_dir).failures

        assert any("not-a-real-chunk-id" in f for f in failures)

    def test_reports_a_stale_source_text_sha(self, run_dir: Path) -> None:
        """A ChunkSet still hashing the raw text means the rewrite never ran."""
        fresh_ids = [c.chunk_id for c in get_chunker(VIEW).chunk(BODY, source_file_id="file-1")]
        write_cep_record(run_dir, "file-1", fresh_ids)

        failures = verify(run_dir).failures

        assert any("source_text_sha256" in f for f in failures)

    def test_reports_a_dangling_bm25_manifest_reference(self, run_dir: Path) -> None:
        fresh_ids = [c.chunk_id for c in get_chunker(VIEW).chunk(BODY, source_file_id="file-1")]
        write_cep_record(run_dir, "file-1", fresh_ids)
        rewrite_chunk_sets(run_dir, dry_run=False)
        directory = run_dir / "retrieve" / "indexes" / "bm25_cep_4k"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"chunk_ids": ["0123456789abcdef"]}))

        failures = verify(run_dir).failures

        assert any("0123456789abcdef" in f for f in failures)

    def test_reports_a_dangling_passage_reference(self, run_dir: Path) -> None:
        fresh_ids = [c.chunk_id for c in get_chunker(VIEW).chunk(BODY, source_file_id="file-1")]
        write_cep_record(run_dir, "file-1", fresh_ids)
        rewrite_chunk_sets(run_dir, dry_run=False)
        write_retrieval_output(run_dir, "retrieve", "bm25", "q0", ["fedcba9876543210"])

        failures = verify(run_dir).failures

        assert any("fedcba9876543210" in f for f in failures)

    def test_ignores_foreign_namespaces_when_checking_for_dangling_refs(
        self, run_dir: Path
    ) -> None:
        fresh_ids = [c.chunk_id for c in get_chunker(VIEW).chunk(BODY, source_file_id="file-1")]
        write_cep_record(run_dir, "file-1", fresh_ids)
        rewrite_chunk_sets(run_dir, dry_run=False)
        write_retrieval_output(run_dir, "retrieve", "atlas_rag", "q0", ["file-9:3"])
        write_retrieval_output(run_dir, "retrieve", "khop_triple", "q1", ["triple:abc123"])

        assert verify(run_dir).failures == []

    def test_reports_the_resolved_fraction(self, run_dir: Path) -> None:
        """The headline number: the fraction, never hard-coded."""
        fresh_ids = [c.chunk_id for c in get_chunker(VIEW).chunk(BODY, source_file_id="file-1")]
        write_cep_record(run_dir, "file-1", fresh_ids)
        rewrite_chunk_sets(run_dir, dry_run=False)

        result = verify(run_dir)

        assert (result.resolved, result.total) == (len(fresh_ids), len(fresh_ids))

    def test_reports_the_fraction_when_pairs_do_not_resolve(self, run_dir: Path) -> None:
        write_cep_record(run_dir, "file-1", ["not-a-real-chunk-id"])
        rewrite_chunk_sets(run_dir, dry_run=False)

        result = verify(run_dir)

        assert (result.resolved, result.total) == (0, 1)
        assert any("0/1" in f for f in result.failures)

    def test_fails_when_the_run_carries_no_cep_pairs(self, run_dir: Path) -> None:
        """Nothing to resolve means the headline check never ran."""
        rewrite_chunk_sets(run_dir, dry_run=False)

        result = verify(run_dir)

        assert result.total == 0
        assert any("no CEP" in f for f in result.failures)


class TestMain:
    """The CLI entry point, against a synthetic run tree only.

    These tests never touch ``results/``. A test that reads a real run would
    couple the suite to 1.5G of data outside the repo and would pass or fail on
    what that data happens to contain rather than on this code.
    """

    def _argv(self, run_dir: Path, *extra: str) -> list[str]:
        """Build an argv pointing the CLI at ``run_dir``."""
        return [
            "migrate_chunk_id_namespace.py",
            "--id",
            run_dir.name,
            "--results-dir",
            str(run_dir.parent),
            *extra,
        ]

    def test_dry_run_reports_without_writing(
        self,
        run_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = run_dir / "chunk" / "outputs" / VIEW / "file-1.json"
        before = path.read_text()
        monkeypatch.setattr(sys, "argv", self._argv(run_dir, "--dry-run"))

        main()

        assert path.read_text() == before
        assert "dry run" in capsys.readouterr().out

    def test_apply_then_verify_reports_consistency(
        self,
        run_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fresh_ids = [c.chunk_id for c in get_chunker(VIEW).chunk(BODY, source_file_id="file-1")]
        write_cep_record(run_dir, "file-1", fresh_ids)

        monkeypatch.setattr(sys, "argv", self._argv(run_dir))
        main()
        capsys.readouterr()

        monkeypatch.setattr(sys, "argv", self._argv(run_dir, "--verify"))
        main()

        out = capsys.readouterr().out
        assert "is consistent in the canonical space" in out
        assert f"{len(fresh_ids)}/{len(fresh_ids)}" in out

    def test_apply_shifts_the_sidecar_and_still_verifies(
        self,
        run_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The whole operator path, sidecar included."""
        fresh_ids = [c.chunk_id for c in get_chunker(VIEW).chunk(BODY, source_file_id="file-1")]
        write_cep_record(run_dir, "file-1", fresh_ids)
        directory = run_dir / "kg" / "outputs"
        directory.mkdir(parents=True)
        sidecar = directory / "passage_offsets.json"
        sidecar.write_text(
            json.dumps(
                {
                    "kg_run_id": "run",
                    "offsets": [
                        {
                            "passage_id": "file-1:0",
                            "source_file_id": "file-1",
                            "start_char": 0,
                            "end_char": 5000,
                            "chunker_id": "atlas_8k",
                        },
                        {
                            "passage_id": "file-1:1",
                            "source_file_id": "file-1",
                            "start_char": 5000,
                            "end_char": 10000,
                            "chunker_id": "atlas_8k",
                        },
                    ],
                    "unmatched": [],
                    "generated_at": "2026-06-24T18:40:49.678102Z",
                }
            )
        )

        monkeypatch.setattr(sys, "argv", self._argv(run_dir))
        main()
        assert "atlas offsets shifted: 2" in capsys.readouterr().out

        spans = [
            (o["start_char"], o["end_char"]) for o in json.loads(sidecar.read_text())["offsets"]
        ]
        assert spans == [(0, 4999), (4999, 9999)]

        monkeypatch.setattr(sys, "argv", self._argv(run_dir, "--verify"))
        main()

        assert "is consistent in the canonical space" in capsys.readouterr().out

    def test_refuses_to_migrate_a_run_that_is_not_a_clone(
        self,
        run_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without ``replicated_from`` the target may be the frozen baseline."""
        write_pipeline_metadata(run_dir, replicated=False)
        path = run_dir / "chunk" / "outputs" / VIEW / "file-1.json"
        before = path.read_text()
        monkeypatch.setattr(sys, "argv", self._argv(run_dir))

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 2
        assert "--allow-original" in capsys.readouterr().out
        assert path.read_text() == before

    def test_allow_original_overrides_the_clone_check(
        self, run_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_pipeline_metadata(run_dir, replicated=False)
        path = run_dir / "chunk" / "outputs" / VIEW / "file-1.json"
        before = path.read_text()
        monkeypatch.setattr(sys, "argv", self._argv(run_dir, "--allow-original"))

        main()

        assert path.read_text() != before

    def test_dry_run_is_not_gated_by_the_clone_check(
        self, run_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The runbook rehearses --dry-run against the frozen baseline."""
        write_pipeline_metadata(run_dir, replicated=False)
        path = run_dir / "chunk" / "outputs" / VIEW / "file-1.json"
        before = path.read_text()
        monkeypatch.setattr(sys, "argv", self._argv(run_dir, "--dry-run"))

        main()

        assert path.read_text() == before

    def test_verify_is_not_gated_by_the_clone_check(
        self, run_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_pipeline_metadata(run_dir, replicated=False)
        monkeypatch.setattr(sys, "argv", self._argv(run_dir, "--verify"))

        with pytest.raises(SystemExit) as excinfo:
            main()

        # Exit 1 is verify's own failure, not the clone gate's exit 2.
        assert excinfo.value.code == 1

    def test_an_abort_in_a_later_step_leaves_nothing_written(
        self, run_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The validating pass runs first, so a sidecar abort spares the ChunkSets."""
        directory = run_dir / "kg" / "outputs"
        directory.mkdir(parents=True)
        (directory / "passage_offsets.json").write_text(
            json.dumps(
                {
                    "kg_run_id": "run",
                    "offsets": [
                        {
                            "passage_id": "file-9:0",
                            "source_file_id": "file-9",
                            "start_char": 10,
                            "end_char": 100,
                            "chunker_id": "atlas_8k",
                        }
                    ],
                    "unmatched": [],
                    "generated_at": "2026-06-24T18:40:49.678102Z",
                }
            )
        )
        path = run_dir / "chunk" / "outputs" / VIEW / "file-1.json"
        before = path.read_text()
        monkeypatch.setattr(sys, "argv", self._argv(run_dir))

        with pytest.raises(ValueError, match="file-9"):
            main()

        assert path.read_text() == before

    def test_verify_exits_one_on_an_unmigrated_run(
        self, run_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guards against a green --verify on a run the rewrite never touched."""
        monkeypatch.setattr(sys, "argv", self._argv(run_dir, "--verify"))

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1

    def test_exits_two_when_the_run_has_no_chunk_stage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["migrate_chunk_id_namespace.py", "--id", "ghost", "--results-dir", str(tmp_path)],
        )

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 2
