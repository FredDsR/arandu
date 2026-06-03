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
    flagged: list[int] = []
    for idx, item in enumerate(ratings):
        present = [r for r in item if r is not None]
        if len(present) < 2:
            continue
        if max(present) - min(present) >= min_spread:
            flagged.append(idx)
    return flagged
