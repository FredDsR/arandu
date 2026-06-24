"""Tests for the chunking batch orchestrator (``run_chunk_batch``).

Focus: the transcription-judge validity filter (``is_valid is False`` records
are skipped so the retrieval corpus matches the QA/KG corpus) and the
``rebuild`` flag (clears stale view outputs + checkpoint for a clean re-run).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from arandu.shared.chunking.batch import run_chunk_batch

if TYPE_CHECKING:
    from pytest import MonkeyPatch


def _write_transcription(
    directory: Path,
    file_id: str,
    text: str = "Test transcription text that is long enough to chunk.",
    is_valid: bool | None = True,
) -> Path:
    """Write a minimal ``EnrichedRecord`` transcription JSON.

    ``is_valid`` is derived from ``validation.passed``. Pass ``None`` to write
    an unjudged record (no ``validation`` payload).
    """
    data: dict[str, Any] = {
        "file_id": file_id,
        "name": f"{file_id}.mp3",
        "mimeType": "audio/mpeg",
        "parents": ["folder"],
        "webContentLink": "https://drive.google.com/test",
        "size_bytes": 1024,
        "duration_milliseconds": 60000,
        "transcription_text": text,
        "detected_language": "pt",
        "language_probability": 0.95,
        "model_id": "whisper-large-v3",
        "compute_device": "cpu",
        "processing_duration_sec": 10.0,
        "transcription_status": "completed",
    }
    if is_valid is not None:
        data["validation"] = {
            "stage_results": {},
            "passed": is_valid,
            "rejected_at": None if is_valid else "heuristic_filter",
        }
    filepath = directory / f"{file_id}_transcription.json"
    filepath.write_text(json.dumps(data))
    return filepath


class TestValidityFilter:
    """run_chunk_batch skips judge-rejected transcriptions."""

    def test_skips_judge_rejected_keeps_valid_and_unjudged(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARANDU_RESULTS_BASE_DIR", str(tmp_path / "results"))
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_transcription(input_dir, "valid", is_valid=True)
        _write_transcription(input_dir, "rejected", is_valid=False)
        _write_transcription(input_dir, "unjudged", is_valid=None)

        result = run_chunk_batch(input_dir=input_dir, views=["cep_4k"], pipeline_id="run1")

        assert result.skipped_invalid == 1
        assert result.sources_processed == 2  # valid + unjudged

        view_dir = Path(result.run_dir) / "outputs" / "cep_4k"
        written = {p.stem for p in view_dir.glob("*.json")}
        assert written == {"valid", "unjudged"}
        assert "rejected" not in written


class TestRebuild:
    """The --rebuild flag clears stale view outputs + checkpoint."""

    def test_rebuild_clears_stale_outputs(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("ARANDU_RESULTS_BASE_DIR", str(tmp_path / "results"))
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_transcription(input_dir, "a", is_valid=True)

        first = run_chunk_batch(input_dir=input_dir, views=["cep_4k"], pipeline_id="run1")
        view_dir = Path(first.run_dir) / "outputs" / "cep_4k"
        stale = view_dir / "stale.json"
        stale.write_text("{}")
        assert stale.exists()

        run_chunk_batch(input_dir=input_dir, views=["cep_4k"], pipeline_id="run1", rebuild=True)

        assert not stale.exists()  # rebuild wiped the view dir
        assert (view_dir / "a.json").exists()  # then re-chunked the valid source
