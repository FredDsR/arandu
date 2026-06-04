"""Shared statistical primitives: percentile and item bootstrap.

Pure-math, dependency-free (no numpy/scipy/statsmodels), mirroring the spirit
of ``shared/rag/analysis/wilson.py``. Promoted out of the agreement module so a
second consumer (the deferred rag-analysis bootstrap CIs) can reuse one tested
implementation instead of forking it.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_CI_LOW_PCTL = 2.5
_CI_HIGH_PCTL = 97.5


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence.

    Matches numpy's default (linear) interpolation.

    Args:
        sorted_values: Ascending-sorted values (non-empty).
        pct: Percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile value.

    Raises:
        ValueError: If ``sorted_values`` is empty.
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile of empty sequence")
    if n == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (n - 1)
    low = int(rank)
    high = min(low + 1, n - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def bootstrap_ci[T](
    items: Sequence[T],
    estimator: Callable[[Sequence[T]], float | None],
    *,
    n_bootstrap: int,
    seed: int,
    low_pct: float = _CI_LOW_PCTL,
    high_pct: float = _CI_HIGH_PCTL,
) -> tuple[float | None, float | None]:
    """Bootstrap a percentile CI by resampling ``items`` with replacement.

    Args:
        items: The units to resample (e.g. label pairs, or per-item rows).
        estimator: Maps a resample to a statistic, or ``None`` if the resample
            is degenerate (those are skipped, not counted as a value).
        n_bootstrap: Number of resamples.
        seed: RNG seed for reproducibility.
        low_pct: Lower percentile (default 2.5).
        high_pct: Upper percentile (default 97.5).

    Returns:
        ``(low, high)`` bounds, or ``(None, None)`` if fewer than two items
        (a bootstrap needs variability to resample) or no resample produced a
        finite statistic.
    """
    n = len(items)
    if n < 2:
        return None, None  # a 1-item bootstrap has no variability to estimate
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        value = estimator(sample)
        if value is not None and math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return None, None
    estimates.sort()
    return percentile(estimates, low_pct), percentile(estimates, high_pct)
