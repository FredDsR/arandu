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
        # n_1=3, n_2=1, n=4 (unobserved scale categories 3,4,5 carry marginal
        # 0 and do not affect the nominal metric). D_o = (1/4)(1+1) = 0.5.
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
        assert r.ci_lower is not None and r.ci_upper is not None
        assert r.coefficient is not None
        assert r.ci_lower <= r.coefficient <= r.ci_upper

    def test_bootstrap_reproducible_with_seed(self) -> None:
        flat_a = [1, 2, 3, 3, 2, 1, 4, 5, 3, 2]
        flat_b = [1, 2, 3, 2, 2, 1, 4, 4, 3, 1]
        r1 = cohen_kappa_weighted(flat_a, flat_b, n_bootstrap=200, seed=7)
        r2 = cohen_kappa_weighted(flat_a, flat_b, n_bootstrap=200, seed=7)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_no_bootstrap_leaves_ci_none(self) -> None:
        r = cohen_kappa_weighted([1, 2, 3], [1, 2, 3])
        assert r.ci_lower is None and r.ci_upper is None

    def test_coefficient_finite(self) -> None:
        r = krippendorff_alpha([[1, 1], [2, 2]], level="ordinal", n_bootstrap=50, seed=1)
        assert r.coefficient is not None and math.isfinite(r.coefficient)

    def test_degenerate_resamples_do_not_pin_ci_high(self) -> None:
        # Mostly-agreeing data with a tail of disagreement: degenerate resamples
        # (collapsing to one category) must be skipped, not counted as 1.0,
        # so the CI is not artificially pinned at the top.
        flat_a = [3, 3, 3, 3, 3, 4, 2]
        flat_b = [3, 3, 3, 3, 3, 2, 4]
        r = cohen_kappa_weighted(flat_a, flat_b, n_bootstrap=500, seed=11)
        assert r.ci_upper is not None and r.ci_upper < 1.0

    def test_single_item_has_no_ci(self) -> None:
        # A 1-item bootstrap cannot estimate variability -> no CI (not a
        # spurious zero-width interval).
        r = cohen_kappa_weighted([2], [4], n_bootstrap=200, seed=1)
        assert r.ci_lower is None and r.ci_upper is None


class TestFixedScaleAndValidation:
    def test_scale_recorded_in_result(self) -> None:
        r = krippendorff_alpha([[1, 1], [2, 2]], scale=(1, 5))
        assert r.scale == (1, 5)

    def test_off_scale_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            cohen_kappa_weighted([1, 7], [1, 2], scale=(1, 5))
        with pytest.raises(ValueError, match="scale"):
            krippendorff_alpha([[1, 9]], scale=(1, 5))
        with pytest.raises(ValueError, match="scale"):
            gwet_ac2([[0, 1]], scale=(1, 5))

    def test_bool_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="bool"):
            cohen_kappa_weighted([True, 2], [1, 2], scale=(1, 5))

    def test_non_finite_label_raises_value_error(self) -> None:
        # NaN/inf must surface as ValueError per the documented contract.
        with pytest.raises(ValueError, match="integer"):
            krippendorff_alpha([[float("nan"), 2]], scale=(1, 5))
        with pytest.raises(ValueError, match="integer"):
            gwet_ac2([[float("inf"), 2]], scale=(1, 5))

    def test_degenerate_scale_rejected_cleanly(self) -> None:
        # Reversed or single-point scales raise a clear scale error, not a
        # confusing per-label "outside scale" error.
        with pytest.raises(ValueError, match="2 points"):
            cohen_kappa_weighted([1, 2], [1, 2], scale=(5, 1))
        with pytest.raises(ValueError, match="2 points"):
            gwet_ac2([[3, 3]], scale=(3, 3))

    def test_ac2_value_independent_of_absent_extreme_category(self) -> None:
        # Same disagreement pattern; presence of an extra (5,5) item that only
        # adds a never-otherwise-used category must not rescale the weights for
        # the rest. With a fixed scale, the shared items' contribution is stable.
        base = [[2, 3], [3, 4], [2, 2], [4, 4]]
        r_narrow = gwet_ac2(base, scale=(1, 5))
        # Widening observed range by adding 1s and 5s should still use span 4.
        r_wide = gwet_ac2([*base, [1, 1], [5, 5]], scale=(1, 5))
        # Not equal (different data), but both computed on span-4 weights:
        # the narrow one must NOT have used span-2 weights. Check a concrete
        # invariant: distance(2,3) under fixed scale is (1/4)^2, so a 1-level
        # miss is mild -> AC2 stays high.
        assert r_narrow.coefficient is not None and r_narrow.coefficient > 0.5
        assert r_wide.coefficient is not None


class TestUndefinedReturnsNone:
    def test_empty_input_is_none(self) -> None:
        assert cohen_kappa_weighted([], []).coefficient is None
        assert krippendorff_alpha([]).coefficient is None
        assert gwet_ac2([]).coefficient is None

    def test_all_missing_is_none(self) -> None:
        assert krippendorff_alpha([[None, None], [None, None]]).coefficient is None
        assert cohen_kappa_weighted([None, None], [None, None]).coefficient is None

    def test_no_variance_is_none_not_zero(self) -> None:
        # Everyone always says "3": no variance -> undefined for all three
        # coefficients (returning None, not 0.0 or a spurious 1.0).
        assert cohen_kappa_weighted([3, 3, 3], [3, 3, 3]).coefficient is None
        assert krippendorff_alpha([[3, 3], [3, 3]]).coefficient is None
        assert gwet_ac2([[3, 3], [3, 3]]).coefficient is None
