"""High-variability item detection for the emic-validity study (spec §6.3).

Items where the human annotators disagree strongly among themselves are
reported separately as a signal of construct fuzziness, not treated as noise:
they count against the reliability of the scale before they count against the
judge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def high_variability_items(
    ratings: Sequence[Sequence[int | None]],
    *,
    min_spread: int = 2,
) -> list[int]:
    """Indices of items whose rating spread is at least ``min_spread``.

    Spread is ``max - min`` over the present ratings of an item. Items with
    fewer than two present ratings are never flagged (no spread to speak of).

    Args:
        ratings: One row per item; each row is the raters' labels (``None`` for
            a missing rating).
        min_spread: Inclusive threshold on ``max - min`` for flagging.

    Returns:
        Sorted list of item indices meeting or exceeding the threshold.
    """
    if min_spread < 1:
        raise ValueError(f"min_spread must be >= 1, got {min_spread}")
    flagged: list[int] = []
    for idx, item in enumerate(ratings):
        present = [r for r in item if r is not None]
        if len(present) < 2:
            continue
        if max(present) - min(present) >= min_spread:
            flagged.append(idx)
    return flagged


def high_variability_rate(
    ratings: Sequence[Sequence[int | None]],
    *,
    min_spread: int = 2,
) -> float | None:
    """Fraction of usable items that are high-variability (spec §6.3).

    The denominator is the number of *usable* items (>= 2 present ratings), so
    the rate is computed over the same items a coefficient would use. Returns
    ``None`` when there are no usable items (rate undefined), mirroring the
    coefficient functions.

    Args:
        ratings: One row per item; each row is the raters' labels.
        min_spread: Inclusive ``max - min`` threshold (see
            :func:`high_variability_items`).

    Returns:
        ``flagged / usable`` in ``[0, 1]``, or ``None`` if no usable items.
    """
    usable = sum(1 for item in ratings if sum(1 for r in item if r is not None) >= 2)
    if usable == 0:
        return None
    return len(high_variability_items(ratings, min_spread=min_spread)) / usable
