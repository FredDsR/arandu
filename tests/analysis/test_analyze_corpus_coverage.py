"""Unit tests for analysis.analyze_corpus_coverage and analysis.plot_corpus_coverage."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

np = pytest.importorskip("numpy")

from analysis.analyze_corpus_coverage import (  # noqa: E402
    fold_text,
    gini,
    map_community,
    sole_cause,
    spearman,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_community_mapping_and_text_folding() -> None:
    assert fold_text("São José / Doquinhas") == "SAO JOSE / DOQUINHAS"
    assert map_community("Barra de Pelotas - RS") == "Barra de Pelotas"
    assert map_community("doquinhas") == "Doquinhas"
    assert map_community("Balsa") == "Balsa"
    assert map_community(None) == "Unspecified"
    assert map_community("Unknown village") == "Unspecified"


def test_sole_cause_classification() -> None:
    assert (
        sole_cause(
            {
                "faithfulness": 0.9,
                "bloom_calibration": 0.9,
                "informativeness": 0.8,
                "self_containedness": 0.8,
            }
        )
        == "pass"
    )
    assert (
        sole_cause(
            {
                "faithfulness": 0.9,
                "bloom_calibration": 0.9,
                "informativeness": 0.2,
                "self_containedness": 0.8,
            }
        )
        == "informativeness"
    )
    assert (
        sole_cause(
            {
                "faithfulness": 0.2,
                "bloom_calibration": 0.2,
                "informativeness": 0.8,
                "self_containedness": 0.8,
            }
        )
        == "2+"
    )


def test_spearman_and_gini() -> None:
    assert math.isclose(spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]), 1.0)
    assert math.isclose(gini([5.0, 5.0, 5.0]), 0.0, abs_tol=1e-9)
    assert math.isnan(gini([0.0, 0.0]))


def test_plot_corpus_coverage_writes_pdf_and_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from analysis.plot_corpus_coverage import concentration_curve, plot_coverage_figure

    x, y = concentration_curve([10.0, 0.0, 10.0])
    assert math.isclose(float(x[-1]), 1.0)
    assert math.isclose(float(y[-1]), 1.0)

    sample = {
        "interviews": {
            "rows": [
                {"file": "f1", "chars": 4000, "useful": 3},
                {"file": "f2", "chars": 1000, "useful": 1},
            ],
            "length_bands": {
                "Q1": {"chars_min": 500, "chars_max": 1000, "yield_per_chunk": 0.5},
                "Q2": {"chars_min": 1000, "chars_max": 4000, "yield_per_chunk": 0.9},
            },
        },
        "graph": {
            "rows": [
                {"file": "f1", "nodes": 40},
                {"file": "f2", "nodes": 10},
            ]
        },
    }
    pdf = tmp_path / "coverage.pdf"
    plot_coverage_figure(sample, pdf)
    assert pdf.exists() and pdf.stat().st_size > 0
    assert pdf.with_suffix(".png").exists()
