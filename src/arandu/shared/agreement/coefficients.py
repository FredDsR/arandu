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

**Fixed scale.** The category set is fixed by the protocol's ordinal scale
(``scale=(1, 5)`` by default), not inferred from the observed labels. This keeps
weighted distances constant across strata (the per-Bloom breakdown in §6) and
across bootstrap resamples, so per-level coefficients are comparable. Labels
outside the scale raise ``ValueError``.

Each function returns an :class:`AgreementResult`; its ``coefficient`` is
``None`` when the statistic is undefined (no usable data, or no expected
disagreement) rather than a misleading ``0.0``. Pass ``n_bootstrap > 0`` and a
``seed`` for a percentile 95% CI.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from arandu.shared.stats import bootstrap_ci

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

Weights = Literal["quadratic", "linear"]
AlphaLevel = Literal["nominal", "ordinal", "interval"]

DEFAULT_SCALE: tuple[int, int] = (1, 5)


class AgreementResult(BaseModel):
    """A coefficient estimate with an optional bootstrap CI.

    Attributes:
        coefficient: Point estimate, or ``None`` when undefined (no usable
            items, or no expected disagreement to correct for).
        ci_lower: Lower 95% bootstrap bound, or ``None``.
        ci_upper: Upper 95% bootstrap bound, or ``None``.
        n_items: Number of items (units) the estimate is based on.
        scale: The fixed ``(min, max)`` ordinal scale used (provenance, so
            callers can confirm two strata were scored comparably).
    """

    coefficient: float | None
    ci_lower: float | None
    ci_upper: float | None
    n_items: int
    scale: tuple[int, int]


def _resolve_scale(scale: tuple[int, int]) -> tuple[list[int], int]:
    """Validate a ``(min, max)`` scale and return ``(categories, span)``.

    The scale must have at least two points (``max > min``); a single-point
    scale carries no notion of agreement. Validating here, before label
    validation, gives a clean error for a reversed/degenerate scale.
    """
    lo, hi = scale
    if hi <= lo:
        raise ValueError(f"scale must have >= 2 points (max > min); got {scale}")
    return list(range(lo, hi + 1)), hi - lo


def _validate_labels(labels: Sequence[int | None], scale: tuple[int, int]) -> None:
    """Reject labels that are not whole integers on the fixed scale."""
    lo, hi = scale
    for v in labels:
        if v is None:
            continue
        if isinstance(v, bool):
            raise ValueError(f"label {v!r} is a bool, not an ordinal value")
        is_int = isinstance(v, int)
        is_whole_float = isinstance(v, float) and math.isfinite(v) and v.is_integer()
        if not (is_int or is_whole_float):
            raise ValueError(f"label {v!r} is not an integer on the {scale} scale")
        if v < lo or v > hi:
            raise ValueError(f"label {v!r} is outside the scale {scale}")


def _validate_units(units: Sequence[Sequence[int | None]], scale: tuple[int, int]) -> None:
    """Validate labels across all items of a units x raters matrix."""
    for unit in units:
        _validate_labels(unit, scale)


def _count_usable_units(units: Sequence[Sequence[int | None]]) -> int:
    """Number of items with at least two present (non-None) ratings."""
    return sum(1 for u in units if sum(1 for r in u if r is not None) >= 2)


def _normalized_disagreement(a: int, b: int, span: int, weights: Weights) -> float:
    """Disagreement in ``[0, 1]`` between two labels on a fixed-span scale.

    Quadratic (default) or linear, normalized by the fixed scale span so the
    metric does not depend on which categories happen to be observed. Distinct
    from the alpha ordinal metric (the ``delta_sq`` defined in ``_alpha_point``).
    """
    if span == 0:
        return 0.0
    if weights == "linear":
        return abs(a - b) / span
    return ((a - b) / span) ** 2


def _finalize[T](
    items: Sequence[T],
    estimator: Callable[[Sequence[T]], float | None],
    n_bootstrap: int,
    seed: int,
    n_items: int,
    scale: tuple[int, int],
) -> AgreementResult:
    """Assemble an AgreementResult from one estimator over ``items``.

    The point estimate is ``estimator(items)`` -- the same callable the
    bootstrap resamples, so the two cannot disagree.
    """
    point = estimator(items)
    ci_lower = ci_upper = None
    if n_bootstrap > 0:
        ci_lower, ci_upper = bootstrap_ci(items, estimator, n_bootstrap=n_bootstrap, seed=seed)
    return AgreementResult(
        coefficient=point,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_items=n_items,
        scale=scale,
    )


# --------------------------------------------------------------------------- #
# Cohen's weighted kappa (pairwise)
# --------------------------------------------------------------------------- #


def _cohen_point(
    pairs: Sequence[tuple[int, int]],
    categories: list[int],
    span: int,
    weights: Weights,
) -> float | None:
    """Weighted Cohen's kappa for (a, b) pairs, or None if undefined."""
    if not pairs:
        return None
    n = len(pairs)

    def dist(a: int, b: int) -> float:
        return _normalized_disagreement(a, b, span, weights)

    d_o = sum(dist(a, b) for a, b in pairs) / n

    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    d_e = sum(
        dist(ci, cj) * (count_a[ci] / n) * (count_b[cj] / n)
        for ci in categories
        for cj in categories
    )

    if d_e == 0:
        return None  # no expected disagreement (a rater is constant) -> undefined
    return 1.0 - d_o / d_e


def cohen_kappa_weighted(
    rater_a: Sequence[int | None],
    rater_b: Sequence[int | None],
    *,
    weights: Weights = "quadratic",
    scale: tuple[int, int] = DEFAULT_SCALE,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> AgreementResult:
    """Weighted Cohen's kappa between two raters on a fixed ordinal scale.

    Args:
        rater_a: Labels from rater A (``None`` entries are dropped pairwise).
        rater_b: Labels from rater B; must be the same length as ``rater_a``.
        weights: ``"quadratic"`` (default) or ``"linear"`` disagreement weights.
        scale: Fixed ``(min, max)`` ordinal scale.
        n_bootstrap: Bootstrap resamples for the CI (0 = no CI).
        seed: RNG seed for the bootstrap.

    Returns:
        AgreementResult; ``coefficient`` is ``None`` when undefined.

    Raises:
        ValueError: If the sequences differ in length, or a label is off-scale.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length")
    categories, span = _resolve_scale(scale)
    _validate_labels(rater_a, scale)
    _validate_labels(rater_b, scale)

    pairs: list[tuple[int, int]] = [
        (a, b) for a, b in zip(rater_a, rater_b, strict=True) if a is not None and b is not None
    ]

    def estimator(sample: Sequence[tuple[int, int]]) -> float | None:
        return _cohen_point(sample, categories, span, weights)

    return _finalize(pairs, estimator, n_bootstrap, seed, len(pairs), scale)


# --------------------------------------------------------------------------- #
# Krippendorff's alpha (multi-rater)
# --------------------------------------------------------------------------- #


def _coincidence_matrix(
    units: Sequence[Sequence[int | None]],
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
    categories: list[int],
    level: AlphaLevel,
) -> float | None:
    """Krippendorff's alpha point estimate, or None if undefined."""
    o = _coincidence_matrix(units)
    n_total = sum(o.values())
    if n_total == 0:
        return None  # no pairable units

    marginals = {c: sum(o.get((c, k), 0.0) for k in categories) for c in categories}

    def delta_sq(c: int, k: int) -> float:
        if level == "nominal":
            return 0.0 if c == k else 1.0
        if level == "interval":
            return float((c - k) ** 2)
        # ordinal (Krippendorff): marginal-based metric.
        lo, hi = (c, k) if c <= k else (k, c)
        between = sum(marginals[g] for g in categories if lo <= g <= hi)
        term = between - (marginals[c] + marginals[k]) / 2.0
        return term * term

    d_o = sum(o.get((c, k), 0.0) * delta_sq(c, k) for c in categories for k in categories)
    d_o /= n_total

    d_e = sum(marginals[c] * marginals[k] * delta_sq(c, k) for c in categories for k in categories)
    d_e /= n_total * (n_total - 1)

    if d_e == 0:
        return None  # no expected disagreement (all present ratings identical)
    return 1.0 - d_o / d_e


def krippendorff_alpha(
    reliability_data: Sequence[Sequence[int | None]],
    *,
    level: AlphaLevel = "ordinal",
    scale: tuple[int, int] = DEFAULT_SCALE,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> AgreementResult:
    """Krippendorff's alpha over items x raters data on a fixed ordinal scale.

    Args:
        reliability_data: One row per item; each row is the raters' labels for
            that item (``None`` for a missing rating). Units with fewer than two
            present ratings contribute nothing.
        level: Distance metric -- ``"ordinal"`` (default; marginal-based),
            ``"interval"`` (squared difference), or ``"nominal"`` (0/1).
        scale: Fixed ``(min, max)`` ordinal scale.
        n_bootstrap: Bootstrap resamples for the CI (0 = no CI).
        seed: RNG seed for the bootstrap.

    Returns:
        AgreementResult; ``coefficient`` is ``None`` when undefined.

    Raises:
        ValueError: If a label is off-scale.
    """
    categories, _ = _resolve_scale(scale)
    _validate_units(reliability_data, scale)

    def estimator(sample: Sequence[Sequence[int | None]]) -> float | None:
        return _alpha_point(sample, categories, level)

    n_items = _count_usable_units(reliability_data)
    return _finalize(reliability_data, estimator, n_bootstrap, seed, n_items, scale)


# --------------------------------------------------------------------------- #
# Gwet's AC2 (multi-rater, weighted)
# --------------------------------------------------------------------------- #


def _ac2_point(
    units: Sequence[Sequence[int | None]],
    categories: list[int],
    span: int,
    weights: Weights,
) -> float | None:
    """Gwet's AC2 point estimate (multi-rater, weighted), or None if undefined."""
    usable = [[r for r in unit if r is not None] for unit in units]
    usable = [u for u in usable if len(u) >= 2]
    if not usable:
        return None

    # No variance (every usable rating identical) -> reliability is undefined,
    # consistent with Cohen/Krippendorff. Without this, AC2 would report a
    # spurious 1.0 for an all-one-category stratum.
    if len({r for u in usable for r in u}) < 2:
        return None

    k = len(categories)
    # Precompute the K x K agreement-weight matrix once (1 on the diagonal).
    agree = {
        (a, b): 1.0 - _normalized_disagreement(a, b, span, weights)
        for a in categories
        for b in categories
    }

    # Observed weighted agreement, averaged over items.
    p_a = 0.0
    for present in usable:
        r_i = len(present)
        counts = Counter(present)
        item_agree = 0.0
        for cat_k, r_ik in counts.items():
            r_star = sum(agree[(cat_k, cat_l)] * counts.get(cat_l, 0) for cat_l in categories)
            item_agree += r_ik * (r_star - 1.0)
        p_a += item_agree / (r_i * (r_i - 1))
    p_a /= len(usable)

    # Chance agreement (Gwet): pi_c = mean proportion in category c over items.
    pi = dict.fromkeys(categories, 0.0)
    for present in usable:
        r_i = len(present)
        counts = Counter(present)
        for c in categories:
            pi[c] += counts.get(c, 0) / r_i
    for c in categories:
        pi[c] /= len(usable)

    # k >= 2 is guaranteed by _resolve_scale (scale has >= 2 points).
    t_w = sum(agree.values())
    p_e = (t_w / (k * (k - 1))) * sum(pi[c] * (1.0 - pi[c]) for c in categories)

    if p_e >= 1.0:
        return None  # undefined
    return (p_a - p_e) / (1.0 - p_e)


def gwet_ac2(
    ratings: Sequence[Sequence[int | None]],
    *,
    weights: Weights = "quadratic",
    scale: tuple[int, int] = DEFAULT_SCALE,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> AgreementResult:
    """Gwet's AC2 over items x raters data on a fixed ordinal scale.

    AC2 redefines chance agreement so it does not inflate when one category
    dominates -- the prevalence paradox that depresses kappa/alpha. Use it as a
    robustness check alongside the primary alpha.

    Args:
        ratings: One row per item; each row is the raters' labels (``None`` for
            missing). Units with fewer than two present ratings are ignored.
        weights: ``"quadratic"`` (default) or ``"linear"`` agreement weights.
        scale: Fixed ``(min, max)`` ordinal scale.
        n_bootstrap: Bootstrap resamples for the CI (0 = no CI).
        seed: RNG seed for the bootstrap.

    Returns:
        AgreementResult; ``coefficient`` is ``None`` when undefined.

    Raises:
        ValueError: If a label is off-scale.
    """
    categories, span = _resolve_scale(scale)
    _validate_units(ratings, scale)

    def estimator(sample: Sequence[Sequence[int | None]]) -> float | None:
        return _ac2_point(sample, categories, span, weights)

    n_items = _count_usable_units(ratings)
    return _finalize(ratings, estimator, n_bootstrap, seed, n_items, scale)
