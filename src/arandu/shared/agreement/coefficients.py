"""Inter-rater agreement coefficients for the emic-validity study (spec §6).

Pure-math implementations from first definitions; no ``statsmodels`` / ``scipy``
/ external reliability package. Three coefficients with complementary roles:

- :func:`krippendorff_alpha` -- multi-rater agreement on an ordinal scale
  (3 anthropologists + the LLM as a 4th rater). The primary metric.
- :func:`cohen_kappa_weighted` -- pairwise (LLM vs each annotator) diagnostic
  with quadratic weights.
- :func:`gwet_ac2` -- robustness check against the prevalence paradox, where a
  skewed label distribution depresses kappa/alpha despite high observed
  agreement.

Data convention (uniform across the three): ratings are a sequence of *items*,
each item a sequence of per-rater labels (integers on the ordinal scale, or
``None`` for a missing rating). Pairwise functions take two equal-length label
sequences instead.

Each function returns an :class:`AgreementResult` carrying the point estimate
and an optional bootstrap 95% CI (resampling items with replacement; pass
``n_bootstrap > 0`` and a ``seed``).
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

Weights = Literal["quadratic", "linear"]
AlphaLevel = Literal["nominal", "ordinal", "interval"]

_CI_LOW_PCTL = 2.5
_CI_HIGH_PCTL = 97.5


@dataclass(frozen=True)
class AgreementResult:
    """A coefficient estimate with an optional bootstrap CI.

    Attributes:
        coefficient: The point estimate.
        ci_low: Lower 95% bootstrap bound, or ``None`` if not requested.
        ci_high: Upper 95% bootstrap bound, or ``None`` if not requested.
        n_items: Number of items (units) the estimate is based on.
    """

    coefficient: float
    ci_low: float | None
    ci_high: float | None
    n_items: int


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def _bootstrap_ci(
    items: Sequence[object],
    estimator: Callable[[Sequence[object]], float | None],
    n_bootstrap: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """Bootstrap a 95% CI by resampling ``items`` with replacement.

    Args:
        items: The per-item units to resample.
        estimator: Maps a resample to a coefficient, or ``None`` if the
            resample is degenerate (skipped).
        n_bootstrap: Number of resamples.
        seed: RNG seed for reproducibility.

    Returns:
        ``(low, high)`` percentile bounds, or ``(None, None)`` if no resample
        yielded a finite coefficient.
    """
    rng = random.Random(seed)
    n = len(items)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        value = estimator(sample)
        if value is not None:
            estimates.append(value)
    if not estimates:
        return None, None
    estimates.sort()
    return _percentile(estimates, _CI_LOW_PCTL), _percentile(estimates, _CI_HIGH_PCTL)


def _distance_sq(a: float, b: float, categories: list[int], weights: Weights) -> float:
    """Squared (quadratic) or absolute (linear) distance between two labels."""
    span = categories[-1] - categories[0]
    if span == 0:
        return 0.0
    if weights == "linear":
        return abs(a - b) / span
    return ((a - b) / span) ** 2


# --------------------------------------------------------------------------- #
# Cohen's weighted kappa (pairwise)
# --------------------------------------------------------------------------- #


def _cohen_point(pairs: Sequence[tuple[int, int]], weights: Weights) -> float | None:
    """Weighted Cohen's kappa for a list of (a, b) label pairs."""
    if not pairs:
        return None
    categories = sorted({c for pair in pairs for c in pair})
    n = len(pairs)

    def dist(a: int, b: int) -> float:
        return _distance_sq(a, b, categories, weights)

    d_o = sum(dist(a, b) for a, b in pairs) / n

    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    d_e = 0.0
    for ci in categories:
        for cj in categories:
            d_e += dist(ci, cj) * (count_a[ci] / n) * (count_b[cj] / n)

    if d_e == 0:
        # No expected disagreement (a rater is constant). Perfect observed
        # agreement -> 1.0; any observed disagreement is undefined -> 0.0.
        return 1.0 if d_o == 0 else 0.0
    return 1.0 - d_o / d_e


def cohen_kappa_weighted(
    rater_a: Sequence[int | None],
    rater_b: Sequence[int | None],
    *,
    weights: Weights = "quadratic",
    n_bootstrap: int = 0,
    seed: int = 0,
) -> AgreementResult:
    """Quadratic-weighted Cohen's kappa between two raters.

    Args:
        rater_a: Labels from rater A (``None`` entries are dropped pairwise).
        rater_b: Labels from rater B; must be the same length as ``rater_a``.
        weights: ``"quadratic"`` (default) or ``"linear"`` disagreement weights.
        n_bootstrap: Bootstrap resamples for the CI (0 = no CI).
        seed: RNG seed for the bootstrap.

    Returns:
        AgreementResult with the kappa estimate and optional CI.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length")

    pairs: list[tuple[int, int]] = [
        (a, b) for a, b in zip(rater_a, rater_b, strict=True) if a is not None and b is not None
    ]

    def estimate(sample: Sequence[object]) -> float | None:
        return _cohen_point(sample, weights)  # type: ignore[arg-type]

    point = _cohen_point(pairs, weights)
    coefficient = 0.0 if point is None else point

    ci_low = ci_high = None
    if n_bootstrap > 0 and pairs:
        ci_low, ci_high = _bootstrap_ci(pairs, estimate, n_bootstrap, seed)

    return AgreementResult(coefficient, ci_low, ci_high, n_items=len(pairs))


# --------------------------------------------------------------------------- #
# Krippendorff's alpha (multi-rater)
# --------------------------------------------------------------------------- #


def _coincidence_matrix(
    units: Sequence[Sequence[int | None]],
    categories: list[int],
) -> dict[tuple[int, int], float]:
    """Krippendorff coincidence matrix from units x raters data.

    Each unit with ``m >= 2`` present ratings contributes ``1/(m-1)`` to every
    ordered pair of its ratings.
    """
    o: dict[tuple[int, int], float] = {}
    for unit in units:
        present = [r for r in unit if r is not None]
        m = len(present)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                key = (present[i], present[j])
                o[key] = o.get(key, 0.0) + 1.0 / (m - 1)
    return o


def _alpha_point(
    units: Sequence[Sequence[int | None]],
    level: AlphaLevel,
) -> float | None:
    """Krippendorff's alpha point estimate for the given metric level."""
    categories = sorted({r for unit in units for r in unit if r is not None})
    if len(categories) < 2:
        # All present ratings identical -> no disagreement possible -> 1.0
        # (if any coincidences exist), else undefined.
        o = _coincidence_matrix(units, categories)
        return 1.0 if o else None

    o = _coincidence_matrix(units, categories)
    n_total = sum(o.values())
    if n_total == 0:
        return None

    marginals = {c: sum(o.get((c, k), 0.0) for k in categories) for c in categories}

    # Ordinal metric needs cumulative marginals between the two categories.
    def delta_sq(c: int, k: int) -> float:
        if level == "nominal":
            return 0.0 if c == k else 1.0
        if level == "interval":
            return float((c - k) ** 2)
        # ordinal (Krippendorff): squared (sum of marginals strictly between
        # plus half the endpoints).
        lo, hi = (c, k) if c <= k else (k, c)
        between = sum(marginals[g] for g in categories if lo <= g <= hi)
        term = between - (marginals[c] + marginals[k]) / 2.0
        return term * term

    d_o = sum(o.get((c, k), 0.0) * delta_sq(c, k) for c in categories for k in categories)
    d_o /= n_total

    d_e = sum(marginals[c] * marginals[k] * delta_sq(c, k) for c in categories for k in categories)
    d_e /= n_total * (n_total - 1)

    if d_e == 0:
        return 1.0 if d_o == 0 else 0.0
    return 1.0 - d_o / d_e


def krippendorff_alpha(
    reliability_data: Sequence[Sequence[int | None]],
    *,
    level: AlphaLevel = "ordinal",
    n_bootstrap: int = 0,
    seed: int = 0,
) -> AgreementResult:
    """Krippendorff's alpha over items x raters data.

    Args:
        reliability_data: One row per item; each row is the raters' labels for
            that item (``None`` for a missing rating). Units with fewer than two
            present ratings contribute nothing.
        level: Distance metric -- ``"ordinal"`` (default; marginal-based),
            ``"interval"`` (squared difference), or ``"nominal"`` (0/1).
        n_bootstrap: Bootstrap resamples for the CI (0 = no CI).
        seed: RNG seed for the bootstrap.

    Returns:
        AgreementResult with the alpha estimate and optional CI.
    """

    def estimate(sample: Sequence[object]) -> float | None:
        return _alpha_point(sample, level)  # type: ignore[arg-type]

    point = _alpha_point(reliability_data, level)
    coefficient = 0.0 if point is None else point

    ci_low = ci_high = None
    if n_bootstrap > 0 and reliability_data:
        ci_low, ci_high = _bootstrap_ci(reliability_data, estimate, n_bootstrap, seed)

    n_items = sum(1 for u in reliability_data if sum(1 for r in u if r is not None) >= 2)
    return AgreementResult(coefficient, ci_low, ci_high, n_items=n_items)


# --------------------------------------------------------------------------- #
# Gwet's AC2 (multi-rater, weighted)
# --------------------------------------------------------------------------- #


def _ac2_point(
    units: Sequence[Sequence[int | None]],
    categories: list[int],
    weights: Weights,
) -> float | None:
    """Gwet's AC2 point estimate (multi-rater, weighted)."""
    if len(categories) < 1:
        return None

    def agree_w(a: int, b: int) -> float:
        # Agreement weight in [0, 1]; 1 on the diagonal.
        return 1.0 - _distance_sq(a, b, categories, weights)

    usable = [[r for r in unit if r is not None] for unit in units]
    usable = [u for u in usable if len(u) >= 2]
    if not usable:
        return None

    k = len(categories)
    # Observed weighted agreement, averaged over items.
    p_a = 0.0
    for present in usable:
        r_i = len(present)
        counts = Counter(present)
        item_agree = 0.0
        for cat_k in categories:
            r_ik = counts.get(cat_k, 0)
            if r_ik == 0:
                continue
            r_star = sum(agree_w(cat_k, cat_l) * counts.get(cat_l, 0) for cat_l in categories)
            item_agree += r_ik * (r_star - 1.0)
        p_a += item_agree / (r_i * (r_i - 1))
    p_a /= len(usable)

    # Chance agreement (Gwet): pi_k = mean proportion in category k over items.
    pi = dict.fromkeys(categories, 0.0)
    for present in usable:
        r_i = len(present)
        counts = Counter(present)
        for c in categories:
            pi[c] += counts.get(c, 0) / r_i
    for c in categories:
        pi[c] /= len(usable)

    if k < 2:
        return 1.0 if p_a == 1.0 else 0.0
    t_w = sum(agree_w(a, b) for a in categories for b in categories)
    p_e = (t_w / (k * (k - 1))) * sum(pi[c] * (1.0 - pi[c]) for c in categories)

    if p_e >= 1.0:
        return 1.0 if p_a >= 1.0 else 0.0
    return (p_a - p_e) / (1.0 - p_e)


def gwet_ac2(
    ratings: Sequence[Sequence[int | None]],
    *,
    weights: Weights = "quadratic",
    n_bootstrap: int = 0,
    seed: int = 0,
) -> AgreementResult:
    """Gwet's AC2 over items x raters data (weighted, multi-rater).

    AC2 redefines chance agreement so it does not inflate when one category
    dominates -- the prevalence paradox that depresses kappa/alpha. Use it as a
    robustness check alongside the primary alpha.

    Args:
        ratings: One row per item; each row is the raters' labels (``None`` for
            missing). Units with fewer than two present ratings are ignored.
        weights: ``"quadratic"`` (default) or ``"linear"`` agreement weights.
        n_bootstrap: Bootstrap resamples for the CI (0 = no CI).
        seed: RNG seed for the bootstrap.

    Returns:
        AgreementResult with the AC2 estimate and optional CI.
    """
    categories = sorted({r for unit in ratings for r in unit if r is not None})

    def estimate(sample: Sequence[object]) -> float | None:
        cats = sorted({r for unit in sample for r in unit if r is not None})  # type: ignore[union-attr]
        return _ac2_point(sample, cats, weights) if cats else None  # type: ignore[arg-type]

    point = _ac2_point(ratings, categories, weights) if categories else None
    coefficient = 0.0 if point is None else point

    ci_low = ci_high = None
    if n_bootstrap > 0 and ratings:
        ci_low, ci_high = _bootstrap_ci(ratings, estimate, n_bootstrap, seed)

    n_items = sum(1 for u in ratings if sum(1 for r in u if r is not None) >= 2)
    return AgreementResult(coefficient, ci_low, ci_high, n_items=n_items)
