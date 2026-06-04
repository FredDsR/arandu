"""Tests for shared statistical primitives (percentile, bootstrap_ci)."""

from __future__ import annotations

import pytest

from arandu.shared.stats import bootstrap_ci, percentile


class TestPercentile:
    def test_single_value(self) -> None:
        assert percentile([4.0], 50) == 4.0

    def test_endpoints(self) -> None:
        data = [0.0, 1.0, 2.0, 3.0, 4.0]
        assert percentile(data, 0) == 0.0
        assert percentile(data, 100) == 4.0

    def test_linear_interpolation(self) -> None:
        # rank = 0.5*(4) = 2.0 -> exact index 2.
        assert percentile([0.0, 10.0, 20.0, 30.0, 40.0], 50) == 20.0
        # rank = 0.25*4 = 1.0 -> index 1.
        assert percentile([0.0, 10.0, 20.0, 30.0, 40.0], 25) == 10.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            percentile([], 50)


class TestBootstrapCI:
    def test_empty_items(self) -> None:
        assert bootstrap_ci([], lambda s: 1.0, n_bootstrap=10, seed=1) == (None, None)

    def test_brackets_constant_estimator(self) -> None:
        lo, hi = bootstrap_ci([1, 2, 3, 4], lambda s: 0.5, n_bootstrap=100, seed=1)
        assert lo == 0.5 and hi == 0.5

    def test_reproducible_with_seed(self) -> None:
        items = list(range(20))

        def est(sample: list[int]) -> float:
            return sum(sample) / len(sample)

        r1 = bootstrap_ci(items, est, n_bootstrap=200, seed=9)
        r2 = bootstrap_ci(items, est, n_bootstrap=200, seed=9)
        assert r1 == r2

    def test_degenerate_estimates_skipped(self) -> None:
        # Estimator returns None for some resamples; CI uses only the rest.
        def est(sample: list[int]) -> float | None:
            return None if all(x == sample[0] for x in sample) else 1.0

        lo, hi = bootstrap_ci([1, 2], est, n_bootstrap=50, seed=3)
        assert lo == 1.0 and hi == 1.0

    def test_all_degenerate_returns_none(self) -> None:
        lo, hi = bootstrap_ci([1, 2, 3], lambda s: None, n_bootstrap=20, seed=1)
        assert lo is None and hi is None
