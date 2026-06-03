"""Tests for the high-variability item detector (spec §6.3)."""

from __future__ import annotations

from arandu.shared.agreement import high_variability_items


class TestHighVariabilityItems:
    def test_flags_spread_at_or_above_threshold(self) -> None:
        # item 0 spread 0, item 1 spread 1, item 2 spread 3 (>=2 flagged).
        ratings = [[3, 3, 3], [3, 4, 3], [1, 4, 2]]
        flagged = high_variability_items(ratings, min_spread=2)
        assert flagged == [2]

    def test_threshold_inclusive(self) -> None:
        ratings = [[2, 4, 3]]  # spread exactly 2
        assert high_variability_items(ratings, min_spread=2) == [0]

    def test_ignores_missing_ratings(self) -> None:
        # spread computed over present ratings only.
        ratings = [[3, None, 3], [1, None, 5]]
        assert high_variability_items(ratings, min_spread=2) == [1]

    def test_single_rating_never_flagged(self) -> None:
        ratings = [[4, None, None]]
        assert high_variability_items(ratings, min_spread=2) == []

    def test_empty_input(self) -> None:
        assert high_variability_items([], min_spread=2) == []

    def test_rejects_non_positive_threshold(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="min_spread"):
            high_variability_items([[3, 3]], min_spread=0)
