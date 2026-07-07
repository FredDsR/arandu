"""Tests for the judge-analysis shared helpers.

Guards the load-bearing logic the threshold/cohort tools rest on: a pair is
approved iff every evaluated criterion meets the threshold, and an unscored /
errored / criterion-less pair never passes.
"""

from __future__ import annotations

from scripts._judge_analysis_common import ANCHORS, approved_ids, snap_to_anchor


def test_approved_requires_all_criteria_at_threshold() -> None:
    scores = {
        "pass": {"faithfulness": 0.75, "bloom_calibration": 1.0},
        "one_below": {"faithfulness": 0.75, "informativeness": 0.5},
        "exact": {"faithfulness": 0.625, "bloom_calibration": 0.625},
    }
    assert approved_ids(scores, 0.625) == {"pass", "exact"}


def test_none_scored_and_empty_never_pass() -> None:
    scores = {
        "errored": {"faithfulness": None, "bloom_calibration": 1.0},
        "empty": {},
        "ok": {"faithfulness": 1.0},
    }
    assert approved_ids(scores, 0.625) == {"ok"}


def test_threshold_in_quantization_gap_is_stable() -> None:
    # Scores are Likert anchors; any threshold in (0.5, 0.75] gives the same set.
    scores = {
        "a": {"c": 0.5},
        "b": {"c": 0.75},
        "c": {"c": 1.0},
    }
    assert approved_ids(scores, 0.625) == approved_ids(scores, 0.7) == {"b", "c"}
    assert approved_ids(scores, 0.5) == {"a", "b", "c"}


def test_snap_to_anchor() -> None:
    assert snap_to_anchor(0.72) == 0.75
    assert snap_to_anchor(0.6) == 0.5
    assert all(snap_to_anchor(a) == a for a in ANCHORS)
