"""Unit tests for analysis.analyze_dense_retrieval."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from analysis.analyze_dense_retrieval import (  # noqa: E402
    compute_hits,
    paired_hit_diff_ci,
    render_markdown,
)


def test_compute_hits_metrics() -> None:
    ranks = np.array([1, 2, 5, 0])
    res = compute_hits(ranks)
    assert res["n"] == 4
    assert math.isclose(res["hit@1"], 0.25)
    assert math.isclose(res["hit@3"], 0.50)
    assert math.isclose(res["hit@10"], 0.75)
    assert math.isclose(res["mrr@10"], (1.0 + 0.5 + 0.2 + 0.0) / 4.0)


def test_paired_hit_diff_ci_and_render_markdown() -> None:
    a = np.array([1, 1, 1, 1, 1])
    b = np.array([0, 0, 0, 0, 0])
    lo, hi = paired_hit_diff_ci(a, b, k=1, n_boot=200, seed=42)
    assert math.isclose(lo, 1.0)
    assert math.isclose(hi, 1.0)

    sample = {
        "run_id": "unit-run",
        "rare_df": 3,
        "n_boot": 200,
        "groups": {
            "useful": {"bm25": {"hit@1": 0.5, "hit@3": 0.7, "hit@10": 0.8, "mrr@10": 0.6, "n": 10}}
        },
        "rare": {
            "share": 0.4,
            "bm25": {
                "with": {"hit@1": 0.7, "hit@10": 0.9},
                "without": {"hit@1": 0.3, "hit@10": 0.7},
            },
        },
        "vs_bm25": {
            "dense_bgem3_256": {
                "useful": {
                    "hit@1": {"diff": 0.1, "ci": [0.01, 0.20]},
                    "hit@10": {"diff": 0.05, "ci": [-0.01, 0.11]},
                }
            }
        },
    }
    md = render_markdown(sample)
    assert "# Retrieval-Only Comparison: BM25 vs. Dense Configurations (unit-run)" in md
    assert "dense_bgem3_256" in md
