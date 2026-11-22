"""Unit tests for analysis.analyze_bloom_yield."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

pd = pytest.importorskip("pandas")

from analysis.analyze_bloom_yield import (  # noqa: E402
    analyze_run,
    applicable_criteria,
    approved_matrix,
    drop_tiny_chunks,
    flag_metadata_echo,
    gini,
    load_cep_pairs,
    rejection_decomposition,
    rejection_reason,
    render_markdown,
    with_usefulness,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_pair(
    chunk_id: str,
    bloom_level: str,
    context_chars: int,
    scores: dict[str, float],
    question: str = "Why did the river rise?",
    answer: str = "Heavy rain flooded the watershed.",
) -> dict:
    passed = all(v >= 0.625 for v in scores.values())
    return {
        "chunk_id": chunk_id,
        "bloom_level": bloom_level,
        "question_type": "open",
        "question": question,
        "answer": answer,
        "context": "x" * context_chars,
        "is_valid": passed,
        "validation": {
            "passed": passed,
            "rejected_at": None if passed else "cep_validation",
            "stage_results": {
                "cep_validation": {
                    "criterion_scores": {
                        name: {"score": val, "threshold": 0.625} for name, val in scores.items()
                    }
                }
            },
        },
    }


def _write_sample_cep_dir(tmp_path: Path) -> Path:
    cep_dir = tmp_path / "cep" / "outputs"
    cep_dir.mkdir(parents=True)
    doc = {
        "source_file_id": "doc-001",
        "source_filename": "interview_01.mp4",
        "source_metadata": {"location": "Barra de Pelotas"},
        "qa_pairs": [
            _make_pair(
                "doc-001:c0",
                "remember",
                1200,
                {"faithfulness": 0.9, "bloom_calibration": 0.9},
            ),
            _make_pair(
                "doc-001:c0",
                "understand",
                1200,
                {
                    "faithfulness": 0.9,
                    "bloom_calibration": 0.8,
                    "informativeness": 0.8,
                    "self_containedness": 0.9,
                },
            ),
            _make_pair(
                "doc-001:c0",
                "analyze",
                1200,
                {
                    "faithfulness": 0.9,
                    "bloom_calibration": 0.9,
                    "informativeness": 0.3,
                    "self_containedness": 0.9,
                },
            ),
            _make_pair(
                "doc-001:c0",
                "evaluate",
                1200,
                {
                    "faithfulness": 0.4,
                    "bloom_calibration": 0.4,
                    "informativeness": 0.8,
                    "self_containedness": 0.8,
                },
                question="What happened in this interview?",
                answer="According to metadata, it happened in May.",
            ),
            # Residual tiny chunk (< 100 chars) should be dropped.
            _make_pair(
                "doc-001:c1",
                "remember",
                40,
                {"faithfulness": 0.9, "bloom_calibration": 0.9},
            ),
        ],
    }
    (cep_dir / "doc-001_cep_qa.json").write_text(json.dumps(doc), encoding="utf-8")
    return cep_dir


def test_drop_tiny_chunks_and_decomposition(tmp_path: Path) -> None:
    cep_dir = _write_sample_cep_dir(tmp_path)
    raw = load_cep_pairs(cep_dir)
    kept, dropped = drop_tiny_chunks(raw, min_chars=100)

    assert len(dropped) == 1
    assert len(kept) == 4

    pairs = with_usefulness(kept)
    assert int(pairs["approved"].sum()) == 2
    assert int(pairs["useful"].sum()) == 1

    matrix = approved_matrix(pairs)
    pairs["reason"] = rejection_reason(pairs, matrix, applicable_criteria(pairs))
    decomp = rejection_decomposition(pairs)
    assert int(decomp.loc["understand", "approved"]) == 1
    assert int(decomp.loc["analyze", "informativeness"]) == 1
    assert int(decomp.loc["evaluate", "multi"]) == 1


def test_metadata_echo_and_gini() -> None:
    df = pd.DataFrame(
        {
            "question": ["Why did it flood?", "What happened nesta entrevista?"],
            "answer": ["Rain fell.", "Conforme os metadados, choveu."],
        }
    )
    flags = flag_metadata_echo(df)
    assert flags.tolist() == [False, True]

    assert math.isclose(gini(pd.Series([10, 10, 10])), 0.0, abs_tol=1e-9)
    assert math.isnan(gini(pd.Series([0, 0])))


def test_analyze_run_end_to_end_and_markdown(tmp_path: Path) -> None:
    cep_dir = _write_sample_cep_dir(tmp_path)
    res = analyze_run(cep_dir, run_id="unit-run")
    assert res["corpus"]["dropped_chunks"] == 1
    assert res["corpus"]["useful_pairs"] == 1
    assert res["metadata_echo"]["all_pairs"] == 1

    md = render_markdown(res)
    assert "# Bloom / CEP Yield Analysis (unit-run)" in md
    assert "Joint Approval by Bloom Level" in md
