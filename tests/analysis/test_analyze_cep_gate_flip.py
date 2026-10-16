"""Tests for the CEP gate-flip report helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from analysis.analyze_cep_gate_flip import (
    build_cohorts,
    flip_criteria,
    load_chunk_owner,
    render_report,
    split_pair_id,
    verdict_kappa,
)
from arandu.shared.chunking.schemas import Chunk, ChunkSet

if TYPE_CHECKING:
    from pathlib import Path


def _chunk(chunk_id: str, file_id: str, view: str, start: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_file_id=file_id,
        chunker_id=view,
        start_char=start,
        end_char=start + 10,
    )


def test_split_pair_id_keeps_file_and_chunk() -> None:
    assert split_pair_id("fileA:abc123:4") == ("fileA", "abc123")
    assert split_pair_id("fileB:none:0") == ("fileB", "none")


def test_cohorts_partition_only_the_shared_pairs() -> None:
    shared = {"a", "b", "c", "d"}
    old_ok = {"a", "b", "only_old"}
    new_ok = {"a", "c", "only_new"}

    cohorts = build_cohorts(old_ok, new_ok, shared)

    assert cohorts == {
        "stable approved": {"a"},
        "newly approved": {"c"},
        "newly rejected": {"b"},
        "stable rejected": {"d"},
    }
    assert set().union(*cohorts.values()) == shared


def test_verdict_kappa_matches_cohen() -> None:
    # p_o = 0.75, p_e = 0.5 * 0.25 + 0.5 * 0.75 = 0.5, kappa = 0.5.
    assert verdict_kappa({"a", "b"}, {"a"}, {"a", "b", "c", "d"}) == pytest.approx(0.5)


def test_verdict_kappa_undefined_when_gates_never_disagree_or_no_pairs() -> None:
    assert verdict_kappa({"a", "b"}, {"a", "b"}, {"a", "b"}) is None
    assert verdict_kappa(set(), set(), set()) is None


def test_flip_criteria_names_crossings_and_judge_errors() -> None:
    old = {"faithfulness": 0.5, "bloom_calibration": 1.0, "informativeness": 0.75}
    new = {"faithfulness": 0.75, "bloom_calibration": 1.0, "informativeness": None}

    assert flip_criteria(old, new, 0.625) == "faithfulness+informativeness (judge error)"
    assert flip_criteria(old, old, 0.625) == "(no criterion: error/absent)"


def test_chunk_owner_covers_every_view_and_skips_bad_files(tmp_path: Path) -> None:
    view_dir = tmp_path / "cep_4k"
    view_dir.mkdir()
    chunk_set = ChunkSet(
        source_file_id="fileA",
        source_filename="a.mp4",
        source_text_sha256="0" * 64,
        views={
            "cep_4k": [_chunk("c1", "fileA", "cep_4k", 0), _chunk("c2", "fileA", "cep_4k", 10)],
            "bm25_512t": [_chunk("b1", "fileA", "bm25_512t", 0)],
        },
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    chunk_set.save(view_dir / "fileA.json")
    (view_dir / "broken.json").write_text("{not json", encoding="utf-8")

    assert load_chunk_owner(tmp_path) == {"c1": "fileA", "c2": "fileA", "b1": "fileA"}


def test_report_refuses_runs_without_shared_pairs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="share no CEP pair"):
        render_report("old", "new", tmp_path, 0.625)
