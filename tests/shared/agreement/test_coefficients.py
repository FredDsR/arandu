"""Tests for inter-rater agreement coefficients (spec §6).

Oracles are hand-computed in the comments so the implementation is pinned to
the definitions, not to a reference library (the module is dependency-free).
"""

from __future__ import annotations

import math

import pytest

from arandu.shared.agreement import (
    cohen_kappa_weighted,
    gwet_ac2,
    krippendorff_alpha,
)


class TestCohenKappaWeighted:
    def test_perfect_agreement_is_one(self) -> None:
        r = cohen_kappa_weighted([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert r.coefficient == pytest.approx(1.0)

    def test_hand_computed_quadratic(self) -> None:
        # A=[1,2,3,3] B=[1,2,3,2]: D_o = 0.25/1.0... see module docstring.
        # D_o = mean (a-b)^2 = (0+0+0+1)/4 = 0.25
        # D_e = sum_ij (i-j)^2 pA_i pB_j = 1.25  -> kappa = 1 - 0.25/1.25 = 0.8
        r = cohen_kappa_weighted([1, 2, 3, 3], [1, 2, 3, 2])
        assert r.coefficient == pytest.approx(0.8)

    def test_systematic_disagreement_negative_or_zero(self) -> None:
        # Raters invert each other on a 2-point scale -> worse than chance.
        r = cohen_kappa_weighted([1, 2, 1, 2], [2, 1, 2, 1])
        assert r.coefficient < 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            cohen_kappa_weighted([1, 2], [1])


class TestKrippendorffAlpha:
    def test_perfect_agreement_is_one(self) -> None:
        # units x raters, all raters agree per unit.
        data = [[3, 3, 3], [1, 1, 1], [5, 5, 5]]
        r = krippendorff_alpha(data, level="ordinal")
        assert r.coefficient == pytest.approx(1.0)

    def test_nominal_hand_computed_zero(self) -> None:
        # units (1,1) and (1,2): coincidences o_11=2, o_12=o_21=1.
        # n_1=3, n_2=1, n=4. D_o = (1/4)(1+1) = 0.5.
        # D_e = (1/(4*3))(3*1 + 1*3) = 0.5. alpha = 1 - 0.5/0.5 = 0.0
        data = [[1, 1], [1, 2]]
        r = krippendorff_alpha(data, level="nominal")
        assert r.coefficient == pytest.approx(0.0, abs=1e-9)

    def test_ordinal_between_nominal_and_perfect(self) -> None:
        # A near-miss (off by one) should hurt ordinal less than a far miss.
        near = krippendorff_alpha([[1, 1], [2, 3], [4, 4], [5, 5]], level="ordinal")
        far = krippendorff_alpha([[1, 1], [2, 5], [4, 4], [5, 5]], level="ordinal")
        assert near.coefficient > far.coefficient

    def test_single_rating_units_ignored(self) -> None:
        # Units with <2 ratings carry no coincidence; perfect on the rest -> 1.
        data = [[3, None], [2, 2], [4, 4]]
        r = krippendorff_alpha(data, level="ordinal")
        assert r.coefficient == pytest.approx(1.0)


class TestGwetAC2:
    def test_perfect_agreement_is_one(self) -> None:
        data = [[1, 1], [5, 5], [3, 3]]
        r = gwet_ac2(data)
        assert r.coefficient == pytest.approx(1.0)

    def test_prevalence_paradox_ac2_exceeds_kappa(self) -> None:
        # 14 items both rate "1", 1 item disagrees. High observed agreement,
        # heavily skewed marginals: kappa collapses, AC2 stays high.
        data = [[1, 1]] * 14 + [[1, 2]]
        ac2 = gwet_ac2(data).coefficient
        flat_a = [row[0] for row in data]
        flat_b = [row[1] for row in data]
        kappa = cohen_kappa_weighted(flat_a, flat_b).coefficient
        assert ac2 > kappa
        assert ac2 > 0.8  # high agreement preserved
        assert kappa < ac2 - 0.3  # the paradox: kappa is depressed


class TestBootstrapCI:
    def test_ci_brackets_point_estimate(self) -> None:
        data = [[1, 1], [2, 2], [3, 2], [4, 4], [5, 5], [3, 3], [2, 1], [4, 5]]
        flat_a = [row[0] for row in data]
        flat_b = [row[1] for row in data]
        r = cohen_kappa_weighted(flat_a, flat_b, n_bootstrap=500, seed=42)
        assert r.ci_low is not None and r.ci_high is not None
        assert r.ci_low <= r.coefficient <= r.ci_high

    def test_bootstrap_reproducible_with_seed(self) -> None:
        flat_a = [1, 2, 3, 3, 2, 1, 4, 5, 3, 2]
        flat_b = [1, 2, 3, 2, 2, 1, 4, 4, 3, 1]
        r1 = cohen_kappa_weighted(flat_a, flat_b, n_bootstrap=200, seed=7)
        r2 = cohen_kappa_weighted(flat_a, flat_b, n_bootstrap=200, seed=7)
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high

    def test_no_bootstrap_leaves_ci_none(self) -> None:
        r = cohen_kappa_weighted([1, 2, 3], [1, 2, 3])
        assert r.ci_low is None and r.ci_high is None

    def test_coefficient_finite(self) -> None:
        r = krippendorff_alpha([[1, 1], [2, 2]], level="ordinal", n_bootstrap=50, seed=1)
        assert math.isfinite(r.coefficient)
