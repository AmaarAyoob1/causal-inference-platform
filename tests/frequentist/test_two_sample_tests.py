"""
Tests for two_sample_tests module.

Three categories:
1. Known-answer tests (cross-checked against scipy)
2. Edge case tests (degenerate inputs)
3. Statistical property tests (Type I error calibration)
"""
import numpy as np
import pytest
from scipy import stats

from core.frequentist.two_sample_tests import welch_t_test, two_proportion_test


# ============================================================
# CATEGORY 1: KNOWN-ANSWER TESTS
# ============================================================

class TestWelchKnownAnswers:
    """Verify against scipy-cross-checked values."""

    def test_matches_scipy_basic(self):
        # Our function calls scipy as ttest_ind(treatment, control, equal_var=False)
        # so the test stat sign convention is: positive = treatment > control
        rng = np.random.default_rng(42)
        control = rng.normal(0, 1, 50)
        treatment = rng.normal(0.5, 1, 50)

        ours = welch_t_test(control, treatment)
        scipy_result = stats.ttest_ind(treatment, control, equal_var=False)

        assert ours.statistic == pytest.approx(scipy_result.statistic, rel=1e-10)
        assert ours.p_value == pytest.approx(scipy_result.pvalue, rel=1e-10)

    def test_positive_effect_positive_statistic(self):
        # Treatment mean > control mean -> t-statistic should be positive
        rng = np.random.default_rng(0)
        control = rng.normal(0, 1, 200)
        treatment = rng.normal(0.5, 1, 200)
        result = welch_t_test(control, treatment)
        assert result.statistic > 0

    def test_zero_effect_high_pvalue(self):
        # Identical distributions should give a non-small p-value (for this seed)
        rng = np.random.default_rng(0)
        control = rng.normal(0, 1, 1000)
        treatment = rng.normal(0, 1, 1000)
        result = welch_t_test(control, treatment)
        assert result.p_value > 0.1

    def test_large_effect_low_pvalue(self):
        rng = np.random.default_rng(0)
        control = rng.normal(0, 1, 200)
        treatment = rng.normal(2, 1, 200)
        result = welch_t_test(control, treatment)
        assert result.p_value < 1e-20


# ============================================================
# CATEGORY 2: EDGE CASES
# ============================================================

class TestWelchEdgeCases:
    """Degenerate inputs. How we handle these IS a design decision."""

    def test_empty_array_raises(self):
        # Currently this will probably error inside scipy or numpy.
        # Whatever happens, it should NOT silently return a "valid" result.
        with pytest.raises(Exception):
            welch_t_test(np.array([]), np.array([1.0, 2.0, 3.0]))

    def test_single_observation_per_group(self):
        # With n=1, variance is undefined (ddof=1 divides by zero).
        # We expect either an exception or NaN.
        try:
            result = welch_t_test(np.array([1.0]), np.array([2.0]))
            # If it didn't raise, the statistic should be NaN
            assert np.isnan(result.statistic) or np.isnan(result.p_value)
        except Exception:
            pass  # raising is also acceptable

    def test_identical_groups_zero_variance(self):
        # Both groups constant and equal. t-stat is 0/0 -> NaN typically.
        result = welch_t_test(
            np.array([5.0, 5.0, 5.0]),
            np.array([5.0, 5.0, 5.0])
        )
        # scipy returns NaN here; our function should propagate that
        assert np.isnan(result.statistic) or result.p_value > 0.99


# ============================================================
# CATEGORY 3: STATISTICAL PROPERTY TESTS (THE IMPORTANT ONES)
# ============================================================

class TestWelchTypeIError:
    """
    Under H0, a valid test at significance level alpha should reject
    ~alpha proportion of the time. Gold standard for validating a test.
    """

    def test_type_one_error_calibrated(self):
        """
        2000 sims under H0 with equal variances. Empirical rejection rate
        should be in [0.035, 0.065] (approx 99% CI around 0.05).
        """
        rng = np.random.default_rng(123)
        alpha = 0.05
        n_sims = 2000
        rejections = 0

        for _ in range(n_sims):
            control = rng.normal(0, 1, 30)
            treatment = rng.normal(0, 1, 30)
            result = welch_t_test(control, treatment)
            if result.p_value < alpha:
                rejections += 1

        rate = rejections / n_sims
        assert 0.035 < rate < 0.065, (
            f"Type I error rate {rate:.4f} outside expected range [0.035, 0.065]. "
            f"Test may be miscalibrated."
        )

    def test_type_one_error_unequal_variances(self):
        """
        2000 sims under H0 with UNEQUAL variances. The whole point of
        Welch's t is handling this — Student's t would FAIL here.
        """
        rng = np.random.default_rng(456)
        alpha = 0.05
        n_sims = 2000
        rejections = 0

        for _ in range(n_sims):
            control = rng.normal(0, 1, 30)    # variance 1
            treatment = rng.normal(0, 4, 30)  # variance 16
            result = welch_t_test(control, treatment)
            if result.p_value < alpha:
                rejections += 1

        rate = rejections / n_sims
        assert 0.035 < rate < 0.065, (
            f"Type I error rate {rate:.4f} under unequal variances. "
            f"Did you accidentally use Student's t instead of Welch's?"
        )


# ============================================================
# TWO PROPORTION TESTS
# ============================================================

class TestTwoProportionKnownAnswers:
    """Sanity checks on direction and basic correctness."""

    def test_direction_positive_lift(self):
        # Treatment converts higher than control -> z should be positive
        result = two_proportion_test(
            control_successes=50, control_total=1000,
            treatment_successes=70, treatment_total=1000,
        )
        assert result.statistic > 0
        assert result.effect_size > 0  # relative lift positive

    def test_direction_negative_lift(self):
        result = two_proportion_test(
            control_successes=70, control_total=1000,
            treatment_successes=50, treatment_total=1000,
        )
        assert result.statistic < 0
        assert result.effect_size < 0

    def test_zero_difference(self):
        result = two_proportion_test(
            control_successes=100, control_total=1000,
            treatment_successes=100, treatment_total=1000,
        )
        assert result.statistic == pytest.approx(0.0, abs=1e-10)
        assert result.p_value == pytest.approx(1.0, abs=1e-10)


class TestTwoProportionTypeIError:
    def test_type_one_error_calibrated(self):
        """
        2000 sims under H0 (same true conversion rate). Should reject ~5%.
        """
        rng = np.random.default_rng(789)
        alpha = 0.05
        n_sims = 2000
        p_true = 0.3
        n = 500
        rejections = 0

        for _ in range(n_sims):
            s_control = rng.binomial(n, p_true)
            s_treatment = rng.binomial(n, p_true)
            result = two_proportion_test(s_control, n, s_treatment, n)
            if result.p_value < alpha:
                rejections += 1

        rate = rejections / n_sims
        assert 0.035 < rate < 0.065, (
            f"Type I error rate {rate:.4f} outside expected range."
        )