"""End-to-end test of the rag-analysis batch driver."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from arandu.shared.rag.analysis.batch import run_rag_analysis_batch

from .conftest import make_answer

if TYPE_CHECKING:
    from pathlib import Path


def _seed_judged_record(
    *,
    base: Path,
    pipeline_id: str,
    arm: str,
    source: str,
    file_stem: str,
    qa_pair_id: str,
    is_answerable: bool,
    abstained: bool,
    abstention_score: float | None,
) -> None:
    """Write a judged AnswerRecord under the canonical layout."""
    target = base / pipeline_id / "judge_answers" / "outputs" / arm / source
    target.mkdir(parents=True, exist_ok=True)
    record = make_answer(
        qa_pair_id=qa_pair_id,
        retriever_id=arm,
        is_answerable=is_answerable,
        abstained=abstained,
        abstention_score=abstention_score,
    )
    record.save(target / f"{file_stem}.json")


def test_missing_judge_outputs_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="judge_answers outputs not found"):
        run_rag_analysis_batch("nope", base_dir=tmp_path)


def test_runs_end_to_end_and_writes_report(tmp_path: Path, cep_outputs_dir: Path) -> None:
    pipeline_id = "run-x"
    base = tmp_path / "results"

    # Reuse the cep_outputs_dir fixture's records via a quick copy into the run.
    target_cep = base / pipeline_id / "cep" / "outputs"
    target_cep.mkdir(parents=True)
    for src in cep_outputs_dir.glob("*.json"):
        (target_cep / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Seed two arms with one judged answer each.
    _seed_judged_record(
        base=base,
        pipeline_id=pipeline_id,
        arm="bm25",
        source="src",
        file_stem="rec_0",
        qa_pair_id="src:chk_00:0",
        is_answerable=True,
        abstained=False,
        abstention_score=0.1,
    )
    _seed_judged_record(
        base=base,
        pipeline_id=pipeline_id,
        arm="atlas_rag",
        source="src",
        file_stem="rec_0",
        qa_pair_id="src:chk_00:1",
        is_answerable=True,
        abstained=False,
        abstention_score=0.1,
    )

    result = run_rag_analysis_batch(pipeline_id, base_dir=base)

    assert result.records_loaded == 2
    assert result.records_unreadable == 0
    assert set(result.arms) == {"bm25", "atlas_rag"}

    payload = json.loads(
        (base / pipeline_id / "analysis" / "outputs" / "report.json").read_text(encoding="utf-8")
    )
    assert payload["pipeline_id"] == pipeline_id
    assert set(payload["joint"].keys()) == {"bm25", "atlas_rag"}
    # Each arm should produce a remember stratum or analyze stratum
    # because the seeded qa_pair_ids match the fixture pairs.
    assert payload["by_bloom"]["bm25"].get("remember") is not None
    assert payload["by_bloom"]["atlas_rag"].get("analyze") is not None


def test_unreadable_records_counted_not_raised(tmp_path: Path) -> None:
    pipeline_id = "run-x"
    base = tmp_path / "results"
    target = base / pipeline_id / "judge_answers" / "outputs" / "bm25" / "src"
    target.mkdir(parents=True)
    (target / "rec_0.json").write_text("not json", encoding="utf-8")

    result = run_rag_analysis_batch(pipeline_id, base_dir=base)
    assert result.records_loaded == 0
    assert result.records_unreadable == 1
