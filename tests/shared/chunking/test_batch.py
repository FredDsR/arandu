"""Tests for the chunking batch orchestrator (``run_chunk_batch``).

Focus: the transcription-judge validity filter (``is_valid is False`` records
are skipped so the retrieval corpus matches the QA/KG corpus) and the
``rebuild`` flag (clears stale view outputs + checkpoint for a clean re-run).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from arandu.shared.chunking.batch import run_chunk_batch

if TYPE_CHECKING:
    from pytest import MonkeyPatch

    from tests.conftest import TranscriptionRecordWriter


class TestValidityFilter:
    """run_chunk_batch skips judge-rejected transcriptions."""

    def test_skips_judge_rejected_keeps_valid_and_unjudged(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        write_transcription_record: TranscriptionRecordWriter,
    ) -> None:
        monkeypatch.setenv("ARANDU_RESULTS_BASE_DIR", str(tmp_path / "results"))
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        write_transcription_record(input_dir, "valid", is_valid=True)
        write_transcription_record(input_dir, "rejected", is_valid=False)
        write_transcription_record(input_dir, "unjudged", is_valid=None)

        result = run_chunk_batch(input_dir=input_dir, views=["cep_4k"], pipeline_id="run1")

        assert result.skipped_invalid == 1
        assert result.sources_processed == 2  # valid + unjudged

        view_dir = Path(result.run_dir) / "outputs" / "cep_4k"
        written = {p.stem for p in view_dir.glob("*.json")}
        assert written == {"valid", "unjudged"}
        assert "rejected" not in written

    def test_all_rejected_leaves_empty_view_dir(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        write_transcription_record: TranscriptionRecordWriter,
    ) -> None:
        """When every source is judge-rejected, the view dir exists but empty.

        A missing dir would crash downstream retrievers resolving the corpus
        path; an empty dir lets them see an empty corpus.
        """
        monkeypatch.setenv("ARANDU_RESULTS_BASE_DIR", str(tmp_path / "results"))
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        write_transcription_record(input_dir, "r1", is_valid=False)
        write_transcription_record(input_dir, "r2", is_valid=False)

        result = run_chunk_batch(input_dir=input_dir, views=["cep_4k"], pipeline_id="run1")

        assert result.skipped_invalid == 2
        assert result.sources_processed == 0
        view_dir = Path(result.run_dir) / "outputs" / "cep_4k"
        assert view_dir.is_dir()
        assert list(view_dir.glob("*.json")) == []


class TestRebuild:
    """The --rebuild flag clears stale view outputs + checkpoint."""

    def test_rebuild_clears_stale_outputs(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        write_transcription_record: TranscriptionRecordWriter,
    ) -> None:
        monkeypatch.setenv("ARANDU_RESULTS_BASE_DIR", str(tmp_path / "results"))
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        write_transcription_record(input_dir, "a", is_valid=True)

        first = run_chunk_batch(input_dir=input_dir, views=["cep_4k"], pipeline_id="run1")
        view_dir = Path(first.run_dir) / "outputs" / "cep_4k"
        stale = view_dir / "stale.json"
        stale.write_text("{}")
        assert stale.exists()

        run_chunk_batch(input_dir=input_dir, views=["cep_4k"], pipeline_id="run1", rebuild=True)

        assert not stale.exists()  # rebuild wiped the view dir
        assert (view_dir / "a.json").exists()  # then re-chunked the valid source
