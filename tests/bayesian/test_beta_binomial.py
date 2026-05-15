"""
Tests for Bayesian Beta-Binomial A/B testing.

Categories:
1. Conjugate update correctness (posterior parameters match the formula)
2. Posterior summary correctness (means, credible intervals)
3. Monte Carlo correctness (P(B>A) sanity, convergence, symmetry)
4. Validation/edge cases
5. Statistical property tests (Bayesian calibration)
"""
import numpy as np
import pytest
from scipy import stats

from core.bayesian.beta_binomial import (
    bayesian_ab_test_beta_binomial,
    BayesianABResult,
)


# ============================================================
# CATEGORY 1: CONJUGATE UPDATE CORRECTNESS
# ============================================================

class TestConjugateUpdate:
    """Posterior parameters must exactly match the closed-form formula."""

    def test_uniform_prior_update(self):
        # Beta(1,1) prior + 10 successes in 100 trials -> Beta(11, 91)
        result = bayesian_ab_test_beta_binomial(
            control_successes=10, control_total=100,
            treatment_successes=10, treatment_total=100,
            prior_alpha=1.0, prior_beta=1.0,
        )
        assert result.alpha_a == 11
        assert result.beta_a == 91
        assert result.alpha_b == 11
        assert result.beta_b == 91

    def test_informative_prior_update(self):
        # Beta(5, 5) prior + 20 successes in 100 trials -> Beta(25, 85)
        result = bayesian_ab_test_beta_binomial(
            control_successes=20, control_total=100,
            treatment_successes=30, treatment_total=100,
            prior_alpha=5.0, prior_beta=5.0,
        )
        assert result.alpha_a == 25
        assert result.beta_a == 85
        assert result.alpha_b == 35
        assert result.beta_b == 75


# ============================================================
# CATEGORY 2: POSTERIOR SUMMARY CORRECTNESS
# ============================================================

class TestPosteriorSummary:
    def test_posterior_mean_matches_formula(self):
        # Mean of Beta(alpha, beta) = alpha / (alpha + beta)
        result = bayesian_ab_test_beta_binomial(
            control_successes=50, control_total=500,
            treatment_successes=60, treatment_total=500,
        )
        expected_mean_a = result.alpha_a / (result.alpha_a + result.beta_a)
        expected_mean_b = result.alpha_b / (result.alpha_b + result.beta_b)
        assert result.posterior_mean_a == pytest.approx(expected_mean_a)
        assert result.posterior_mean_b == pytest.approx(expected_mean_b)

    def test_credible_interval_matches_beta_ppf(self):
        result = bayesian_ab_test_beta_binomial(
            control_successes=50, control_total=500,
            treatment_successes=60, treatment_total=500,
            credible_level=0.95,
        )
        expected_lo = stats.beta.ppf(0.025, result.alpha_a, result.beta_a)
        expected_hi = stats.beta.ppf(0.975, result.alpha_a, result.beta_a)
        assert result.credible_interval_a[0] == pytest.approx(expected_lo)
        assert result.credible_interval_a[1] == pytest.approx(expected_hi)

    def test_with_strong_data_posterior_near_empirical(self):
        # With lots of data, posterior mean ~ empirical rate
        result = bayesian_ab_test_beta_binomial(
            control_successes=1000, control_total=10_000,
            treatment_successes=1100, treatment_total=10_000,
        )
        # Posterior should be very close to 0.10 and 0.11 (data dominates prior)
        assert abs(result.posterior_mean_a - 0.10) < 0.001
        assert abs(result.posterior_mean_b - 0.11) < 0.001


# ============================================================
# CATEGORY 3: MONTE CARLO CORRECTNESS
# ============================================================

class TestMonteCarloProperty:
    def test_identical_arms_gives_near_half(self):
        """Identical data on both arms -> P(B>A) ~ 0.5."""
        result = bayesian_ab_test_beta_binomial(
            control_successes=50, control_total=500,
            treatment_successes=50, treatment_total=500,
            n_mc_samples=100_000,
        )
        # With 100k samples, sampling SE is small. Allow a wide band.
        assert 0.48 < result.probability_b_beats_a < 0.52

    def test_strongly_better_b_gives_near_one(self):
        result = bayesian_ab_test_beta_binomial(
            control_successes=50, control_total=1000,
            treatment_successes=200, treatment_total=1000,  # huge lift
        )
        assert result.probability_b_beats_a > 0.999

    def test_strongly_better_a_gives_near_zero(self):
        result = bayesian_ab_test_beta_binomial(
            control_successes=200, control_total=1000,
            treatment_successes=50, treatment_total=1000,
        )
        assert result.probability_b_beats_a < 0.001

    def test_symmetry(self):
        """Swapping arms should give complementary probabilities."""
        forward = bayesian_ab_test_beta_binomial(
            control_successes=100, control_total=1000,
            treatment_successes=120, treatment_total=1000,
            random_state=42,
        )
        backward = bayesian_ab_test_beta_binomial(
            control_successes=120, control_total=1000,
            treatment_successes=100, treatment_total=1000,
            random_state=42,
        )
        # P(B>A) in one direction + P(B>A) in the other should ~ 1
        # (modulo ties, which are zero-probability for continuous posteriors)
        assert abs(forward.probability_b_beats_a + backward.probability_b_beats_a - 1.0) < 0.005


# ============================================================
# CATEGORY 4: VALIDATION
# ============================================================

class TestValidation:
    def test_negative_successes_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            bayesian_ab_test_beta_binomial(
                control_successes=-1, control_total=100,
                treatment_successes=10, treatment_total=100,
            )

    def test_successes_exceeding_trials_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            bayesian_ab_test_beta_binomial(
                control_successes=150, control_total=100,
                treatment_successes=10, treatment_total=100,
            )

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="positive"):
            bayesian_ab_test_beta_binomial(
                control_successes=0, control_total=0,
                treatment_successes=10, treatment_total=100,
            )

    def test_invalid_prior_raises(self):
        with pytest.raises(ValueError, match="prior"):
            bayesian_ab_test_beta_binomial(
                control_successes=10, control_total=100,
                treatment_successes=10, treatment_total=100,
                prior_alpha=-1.0,
            )

    def test_too_few_mc_samples_raises(self):
        with pytest.raises(ValueError, match="MC samples"):
            bayesian_ab_test_beta_binomial(
                control_successes=10, control_total=100,
                treatment_successes=10, treatment_total=100,
                n_mc_samples=10,
            )


# ============================================================
# CATEGORY 5: STATISTICAL PROPERTY — BAYESIAN CALIBRATION
# ============================================================

class TestBayesianCalibration:
    """
    A more subtle check: 95% credible intervals from a correctly
    specified Bayesian model should contain the true parameter ~95%
    of the time (averaged over the prior). This is the Bayesian
    analog of frequentist coverage.

    We use a flat prior, so any prior-consistent simulation works.
    """

    def test_credible_interval_coverage(self):
        rng = np.random.default_rng(2025)
        n_sims = 500
        n_trials = 200
        coverage = 0

        for _ in range(n_sims):
            # Sample a "true" rate from the prior Beta(1, 1) = Uniform(0, 1)
            true_p = rng.uniform(0, 1)
            # Simulate data at that rate
            successes = rng.binomial(n_trials, true_p)
            # Run the test (we only care about arm A's CI)
            result = bayesian_ab_test_beta_binomial(
                control_successes=successes, control_total=n_trials,
                treatment_successes=successes, treatment_total=n_trials,
                prior_alpha=1.0, prior_beta=1.0,
                n_mc_samples=2000,  # small to keep test fast
            )
            lo, hi = result.credible_interval_a
            if lo <= true_p <= hi:
                coverage += 1

        rate = coverage / n_sims
        # With 500 sims, the 99% CI on the empirical coverage rate is
        # roughly 0.95 +/- 0.025. Use [0.92, 0.98] to be safe.
        assert 0.92 < rate < 0.98, (
            f"Credible interval coverage {rate:.3f} far from 0.95. "
            f"Bayesian model may be miscalibrated."
        )