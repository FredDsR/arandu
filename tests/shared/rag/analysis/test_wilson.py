"""Tests for the Wilson 95% confidence interval helper."""

from __future__ import annotations

import math

import pytest

from arandu.shared.rag.analysis.wilson import wilson_ci


class TestWilsonCI:
    def test_zero_denominator_returns_zero_band(self) -> None:
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_all_successes_upper_at_one(self) -> None:
        lower, upper = wilson_ci(10, 10)
        assert lower < upper == 1.0 or upper == pytest.approx(1.0)
        assert lower < 1.0
        assert lower > 0.5

    def test_no_successes_lower_at_zero(self) -> None:
        lower, upper = wilson_ci(0, 10)
        assert lower == 0.0
        assert 0.0 < upper < 1.0

    def test_half_proportion_centers_on_half(self) -> None:
        lower, upper = wilson_ci(50, 100)
        assert lower < 0.5 < upper
        assert math.isclose(lower + upper, 1.0, abs_tol=0.001)

    def test_negative_inputs_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            wilson_ci(-1, 10)
        with pytest.raises(ValueError, match="must be >= 0"):
            wilson_ci(0, -1)

    def test_excess_successes_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be <="):
            wilson_ci(11, 10)
