"""Tests for ``arandu.kg.passage_offsets`` — atlas-rag passage offset mapper."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from arandu.kg.passage_offsets import (
    PassageOffset,
    PassageOffsetSidecar,
    link_passages,
)

if TYPE_CHECKING:
    from pathlib import Path


_HEADER = (
    "[Contexto da Entrevista]\n"
    "Local: BARRA DE PELOTAS\n"
    "Data: 18-07-2025\n"
    "Contexto: Entrevista D. Celia\n\n"
    "[Transcrição]\n"
)


def _write_enriched_record(out_dir: Path, file_id: str, transcription_text: str) -> None:
    """Write a minimal EnrichedRecord JSON for the linker to load."""
    payload = {
        "gdrive_id": file_id,
        "name": f"{file_id}.mp4",
        "mimeType": "video/mp4",
        "parents": ["folder"],
        "webContentLink": "https://drive.google.com/test",
        "size_bytes": 1024,
        "duration_milliseconds": 60000,
        "transcription_text": transcription_text,
        "detected_language": "pt",
        "language_probability": 0.95,
        "model_id": "whisper-large-v3",
        "compute_device": "cpu",
        "processing_duration_sec": 30.5,
        "transcription_status": "completed",
    }
    (out_dir / f"{file_id}.json").write_text(json.dumps(payload))


def _write_kg_extraction_jsonl(out_dir: Path, records: list[dict]) -> None:
    """Write atlas-rag-style JSONL with one record per line."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "qwen3:14b_transcriptions_out_1_in_1.json"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


@pytest.fixture
def kg_run_fixture(tmp_path: Path) -> tuple[Path, str]:
    """Build a `results/<pipeline_id>/{transcription,kg}/outputs/...` fixture.

    Returns:
        (base_dir, pipeline_id) ready for ``link_passages``.
    """
    pipeline_id = "test_run_001"
    base = tmp_path / "results"
    tr_out = base / pipeline_id / "transcription" / "outputs"
    tr_out.mkdir(parents=True)
    kg_ext = base / pipeline_id / "kg" / "outputs" / "atlas_output" / "kg_extraction"

    # File A — single atlas chunk
    text_a = "Esta é a transcrição A. O rio Uruguai subiu três metros em poucas horas."
    _write_enriched_record(tr_out, "src_a", text_a)

    # File B — split into two atlas chunks; chunk_2 starts mid-text
    text_b_pt1 = "Parte 1 da transcrição B. Contém o primeiro segmento. "
    text_b_pt2 = "Parte 2 da transcrição B. Outro segmento separado."
    text_b = text_b_pt1 + text_b_pt2
    _write_enriched_record(tr_out, "src_b", text_b)

    # File C — included in transcription but missing from kg_extraction (orphan source case)
    _write_enriched_record(tr_out, "src_c", "Texto solitário sem extração.")

    records = [
        {"id": "src_a", "original_text": _HEADER + text_a, "metadata": {"lang": "pt"}},
        {"id": "src_b", "original_text": _HEADER + text_b_pt1, "metadata": {"lang": "pt"}},
        {"id": "src_b", "original_text": _HEADER + text_b_pt2, "metadata": {"lang": "pt"}},
    ]
    _write_kg_extraction_jsonl(kg_ext, records)
    return base, pipeline_id


class TestPassageOffsetSchemas:
    def test_passage_offset_defaults_chunker_id(self) -> None:
        po = PassageOffset(passage_id="src_a:0", source_file_id="src_a", start_char=0, end_char=10)
        assert po.chunker_id == "atlas_8k"

    def test_sidecar_roundtrip(self, tmp_path: Path) -> None:
        sidecar = PassageOffsetSidecar(
            kg_run_id="test_run_001",
            offsets=[
                PassageOffset(
                    passage_id="src_a:0", source_file_id="src_a", start_char=0, end_char=10
                )
            ],
            unmatched=["src_x:0"],
            generated_at=datetime.now(UTC),
        )
        path = tmp_path / "passage_offsets.json"
        sidecar.save(path)
        loaded = PassageOffsetSidecar.load(path)
        assert loaded.kg_run_id == sidecar.kg_run_id
        assert loaded.offsets[0].passage_id == "src_a:0"
        assert loaded.unmatched == ["src_x:0"]


class TestLinkPassages:
    def test_writes_sidecar_at_default_kg_outputs_path(
        self, kg_run_fixture: tuple[Path, str]
    ) -> None:
        base, pid = kg_run_fixture
        sidecar = link_passages(pipeline_id=pid, base_dir=base)

        expected_path = base / pid / "kg" / "outputs" / "passage_offsets.json"
        assert expected_path.exists()

        loaded = PassageOffsetSidecar.load(expected_path)
        assert loaded == sidecar

    def test_offsets_anchor_at_correct_start_char(self, kg_run_fixture: tuple[Path, str]) -> None:
        base, pid = kg_run_fixture
        sidecar = link_passages(pipeline_id=pid, base_dir=base)

        # src_a: single chunk, whole transcription → start_char=0
        a = next(o for o in sidecar.offsets if o.passage_id == "src_a:0")
        assert a.source_file_id == "src_a"
        assert a.start_char == 0
        assert a.end_char > a.start_char

        # src_b: two chunks; the second must start at the position where part 2 begins
        b0 = next(o for o in sidecar.offsets if o.passage_id == "src_b:0")
        b1 = next(o for o in sidecar.offsets if o.passage_id == "src_b:1")
        assert b0.start_char == 0
        # Part 2 starts where part 1 ends — sequential, non-overlapping
        assert b1.start_char == b0.end_char

    def test_passage_id_synthesis_when_atlas_does_not_provide_one(
        self, kg_run_fixture: tuple[Path, str]
    ) -> None:
        # atlas-rag emits per-record `id` (the source file_id), no passage_id.
        # The linker synthesises one as `<file_id>:<chunk_index>` based on JSONL order.
        base, pid = kg_run_fixture
        sidecar = link_passages(pipeline_id=pid, base_dir=base)

        ids = {o.passage_id for o in sidecar.offsets}
        assert ids == {"src_a:0", "src_b:0", "src_b:1"}

    def test_header_stripping_handles_atlas_injected_prelude(
        self, kg_run_fixture: tuple[Path, str]
    ) -> None:
        # The kg_extraction record's `original_text` starts with the atlas-rag
        # `[Contexto da Entrevista]...[Transcrição]\n` header that does NOT exist
        # in EnrichedRecord.transcription_text. Successful anchoring proves the
        # linker stripped the header before searching.
        base, pid = kg_run_fixture
        sidecar = link_passages(pipeline_id=pid, base_dir=base)

        # All passages anchored (no unmatched).
        assert sidecar.unmatched == []
        assert len(sidecar.offsets) == 3

    def test_unmatched_passages_recorded_for_audit(self, tmp_path: Path) -> None:
        # If a passage's chunk text cannot be found in the source transcription,
        # surface the passage_id under `unmatched` instead of silently dropping it.
        pid = "test_run_x"
        base = tmp_path / "results"
        tr_out = base / pid / "transcription" / "outputs"
        tr_out.mkdir(parents=True)
        _write_enriched_record(tr_out, "src_a", "Some legitimate transcription.")

        kg_ext = base / pid / "kg" / "outputs" / "atlas_output" / "kg_extraction"
        # Passage text doesn't appear in the EnrichedRecord at all.
        _write_kg_extraction_jsonl(
            kg_ext,
            [
                {
                    "id": "src_a",
                    "original_text": _HEADER + "Text that does not exist in source.",
                    "metadata": {"lang": "pt"},
                }
            ],
        )

        sidecar = link_passages(pipeline_id=pid, base_dir=base)
        assert sidecar.unmatched == ["src_a:0"]
        assert sidecar.offsets == []

    def test_whitespace_normalized_fallback(self, tmp_path: Path) -> None:
        # Atlas-rag occasionally re-flows whitespace. Exact `.find()` would miss
        # such cases; the linker must retry with whitespace-normalised matching
        # (per spec §3.8).
        pid = "test_run_x"
        base = tmp_path / "results"
        tr_out = base / pid / "transcription" / "outputs"
        tr_out.mkdir(parents=True)
        _write_enriched_record(tr_out, "src_a", "Linha 1.\nLinha 2.\nLinha 3.")

        kg_ext = base / pid / "kg" / "outputs" / "atlas_output" / "kg_extraction"
        # Chunk text uses single spaces where the source has newlines — exact find fails,
        # whitespace-normalised find should succeed.
        _write_kg_extraction_jsonl(
            kg_ext,
            [
                {
                    "id": "src_a",
                    "original_text": _HEADER + "Linha 1. Linha 2. Linha 3.",
                    "metadata": {"lang": "pt"},
                }
            ],
        )

        sidecar = link_passages(pipeline_id=pid, base_dir=base)
        assert sidecar.unmatched == []
        assert len(sidecar.offsets) == 1
        # Offsets cover the source span (start at 0; end at len of source).
        assert sidecar.offsets[0].start_char == 0
        assert sidecar.offsets[0].end_char == len("Linha 1.\nLinha 2.\nLinha 3.")

    def test_kg_run_id_recorded_in_sidecar(self, kg_run_fixture: tuple[Path, str]) -> None:
        base, pid = kg_run_fixture
        sidecar = link_passages(pipeline_id=pid, base_dir=base)
        assert sidecar.kg_run_id == pid

    def test_orphan_source_file_id_skipped_with_warning(
        self, kg_run_fixture: tuple[Path, str]
    ) -> None:
        # src_c has an EnrichedRecord but no atlas-rag extraction. That's not
        # an error — the KG simply didn't index it. Sidecar should not contain
        # src_c entries.
        base, pid = kg_run_fixture
        sidecar = link_passages(pipeline_id=pid, base_dir=base)
        sources = {o.source_file_id for o in sidecar.offsets}
        assert "src_c" not in sources

    def test_handles_production_transcription_suffix(self, tmp_path: Path) -> None:
        # Production runs (e.g. results/test-kg-04/) name files as
        # `<file_id>_transcription.json` rather than `<file_id>.json`.
        # The linker must accept either form.
        pid = "prod_like_run"
        base = tmp_path / "results"
        tr_out = base / pid / "transcription" / "outputs"
        tr_out.mkdir(parents=True)

        text = "Production-style transcription text."
        prod_payload = {
            "gdrive_id": "src_a",
            "name": "src_a.mp4",
            "mimeType": "video/mp4",
            "parents": ["folder"],
            "webContentLink": "https://drive.google.com/test",
            "size_bytes": 1024,
            "duration_milliseconds": 60000,
            "transcription_text": text,
            "detected_language": "pt",
            "language_probability": 0.95,
            "model_id": "whisper-large-v3",
            "compute_device": "cpu",
            "processing_duration_sec": 30.5,
            "transcription_status": "completed",
        }
        (tr_out / "src_a_transcription.json").write_text(json.dumps(prod_payload))

        kg_ext = base / pid / "kg" / "outputs" / "atlas_output" / "kg_extraction"
        _write_kg_extraction_jsonl(
            kg_ext,
            [{"id": "src_a", "original_text": _HEADER + text, "metadata": {"lang": "pt"}}],
        )

        sidecar = link_passages(pipeline_id=pid, base_dir=base)
        assert sidecar.unmatched == []
        assert len(sidecar.offsets) == 1
        assert sidecar.offsets[0].source_file_id == "src_a"
