"""Tests for report.json + tables.md emission."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from arandu.shared.rag.analysis.metrics import aggregate_arm
from arandu.shared.rag.analysis.report import AnalysisReport, write_report

from .conftest import make_answer

if TYPE_CHECKING:
    from pathlib import Path


def test_write_report_produces_both_files(tmp_path: Path) -> None:
    records = [
        make_answer(
            qa_pair_id="r:c:0",
            retriever_id="bm25",
            is_answerable=True,
            abstained=False,
        ),
        make_answer(
            qa_pair_id="r:c:1",
            retriever_id="bm25",
            is_answerable=False,
            abstained=True,
            abstention_score=0.9,
        ),
    ]
    joint = {"bm25": aggregate_arm("bm25", records)}
    report = AnalysisReport(pipeline_id="run-x", joint=joint)

    json_path, md_path = write_report(tmp_path, report)
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["pipeline_id"] == "run-x"
    assert "bm25" in payload["joint"]

    md = md_path.read_text(encoding="utf-8")
    assert "Table 1" in md
    assert "bm25" in md
    assert "run-x" in md


def test_render_handles_bloom_and_question_type_tables(tmp_path: Path) -> None:
    records = [
        make_answer(qa_pair_id="r:c:0", retriever_id="bm25"),
    ]
    metrics = aggregate_arm("bm25", records, slice_name="bloom=remember")
    report = AnalysisReport(
        pipeline_id="run-x",
        joint={"bm25": metrics},
        by_bloom={"bm25": {"remember": metrics}},
        by_question_type={"bm25": {"factual": metrics}},
    )

    _, md_path = write_report(tmp_path, report)
    md = md_path.read_text(encoding="utf-8")
    assert "Table 2" in md
    assert "Table 3" in md
    assert "remember" in md
    assert "factual" in md
