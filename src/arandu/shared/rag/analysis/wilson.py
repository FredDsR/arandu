"""Wilson 95% confidence interval for a binary proportion (spec §8.3).

Pure-math implementation; no ``statsmodels`` / ``scipy`` dependency. The
spec calls for ``statsmodels.stats.proportion.proportion_confint(method="wilson")``
which uses exactly this closed form - keeping it inline avoids dragging
in a heavy stats dep for a 6-line formula.

Wilson is preferred over the normal-approximation CI for small ``n``
and for proportions near 0 or 1, where the normal approximation breaks
down (the thesis cares about both - abstention rates can be near zero;
strata can be small per Bloom level).
"""

from __future__ import annotations

from math import sqrt

# z-score for 95% confidence (two-sided normal).
_Z_95 = 1.96


def wilson_ci(k: int, n: int, *, z: float = _Z_95) -> tuple[float, float]:
    """Compute the Wilson confidence interval for ``k`` successes out of ``n``.

    Args:
        k: Number of successes (must satisfy ``0 <= k <= n``).
        n: Total trials. ``n == 0`` returns ``(0.0, 0.0)`` defensively
            (a stratum with no items can't anchor a CI; surfaces as
            an empty band in the report instead of a ZeroDivisionError).
        z: z-score for the desired confidence level. Defaults to 1.96
            (95% confidence).

    Returns:
        ``(lower, upper)`` bounds in ``[0.0, 1.0]``. Returns
        ``(0.0, 0.0)`` for ``n == 0``.

    Raises:
        ValueError: If ``k < 0``, ``n < 0``, or ``k > n``.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if k > n:
        raise ValueError(f"k ({k}) must be <= n ({n})")
    if n == 0:
        return 0.0, 0.0

    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    return lower, upper
