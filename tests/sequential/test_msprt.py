"""
Tests for mSPRT module.

Categories:
1. Basic correctness — single-point computation matches math
2. Sequential behavior — stops correctly on real effects, continues on null
3. Statistical property — Type I error stays bounded under peeking (the
   punchline test — this is THE thing mSPRT claims to do)
4. Validation
"""
import numpy as np
import pytest

from core.sequential.msprt import (
    msprt_beta_binomial,
    msprt_sequential,
    mSPRTResult,
)


# ============================================================
# CATEGORY 1: SINGLE-POINT CORRECTNESS
# ============================================================

class TestBasicCorrectness:
    def test_identical_arms_no_rejection(self):
        """
        
        Equal data on both arms: no evidence of difference, so we should NOT
        reject. Log-LR should be negative (data is more consistent with the
        'arms are similar' hypothesis than the 'arms differ' hypothesis) and
        well below the rejection threshold.
        
        """
        result = msprt_beta_binomial(
            control_successes=50, control_total=500,
            treatment_successes=50, treatment_total=500,
        )
        assert not result.reject_null
        # Identical data favors H0; log-LR should be substantially below the
        # rejection threshold log(1/alpha) ≈ 3.0.
        assert result.log_likelihood_ratio < result.log_rejection_threshold - 1.0
        

    def test_huge_effect_rejects(self):
        """A massive lift: should reject decisively."""
        result = msprt_beta_binomial(
            control_successes=50, control_total=1000,
            treatment_successes=300, treatment_total=1000,
        )
        assert result.reject_null
        # Log-LR should be very large
        assert result.log_likelihood_ratio > 10

    def test_log_lambda_matches_exp_lambda(self):
        """The reported lambda should equal exp(log_lambda)."""
        result = msprt_beta_binomial(
            control_successes=100, control_total=1000,
            treatment_successes=120, treatment_total=1000,
        )
        assert result.likelihood_ratio == pytest.approx(
            np.exp(result.log_likelihood_ratio), rel=1e-6
        )

    def test_threshold_matches_alpha(self):
        """Rejection threshold should be 1/alpha."""
        result = msprt_beta_binomial(
            control_successes=50, control_total=500,
            treatment_successes=50, treatment_total=500,
            alpha_signif=0.05,
        )
        assert result.rejection_threshold == pytest.approx(20.0)

        result_strict = msprt_beta_binomial(
            control_successes=50, control_total=500,
            treatment_successes=50, treatment_total=500,
            alpha_signif=0.01,
        )
        assert result_strict.rejection_threshold == pytest.approx(100.0)


# ============================================================
# CATEGORY 2: SEQUENTIAL BEHAVIOR
# ============================================================

class TestSequentialBehavior:
    def test_real_effect_stops_early(self):
        """With a large real effect, mSPRT should reject well before n_max."""
        rng = np.random.default_rng(0)
        n_max = 5000
        control = rng.binomial(1, 0.10, n_max)
        treatment = rng.binomial(1, 0.20, n_max)  # huge lift

        history = msprt_sequential(control, treatment, check_every=50)
        final = history[-1]
        assert final.reject_null
        # Should stop well before n_max
        assert final.n_a < n_max
        assert final.n_a < 2000  # large effect -> very early stop

    def test_no_effect_does_not_reject(self):
        """Under H0, mSPRT should NOT reject, even with many peeks."""
        rng = np.random.default_rng(7)
        n_max = 3000
        control = rng.binomial(1, 0.10, n_max)
        treatment = rng.binomial(1, 0.10, n_max)

        history = msprt_sequential(control, treatment, check_every=50)
        final = history[-1]
        # This is probabilistic — under H0 there's still alpha=5% chance
        # of rejecting somewhere in the sequence. We pick a seed where
        # it doesn't reject. The full property test is in Category 3.
        assert not final.reject_null
        assert final.n_a == n_max  # ran to completion


# ============================================================
# CATEGORY 3: TYPE I ERROR UNDER PEEKING (THE PUNCHLINE)
# ============================================================

class TestTypeIErrorUnderPeeking:
    """
    The defining claim of mSPRT: even with continuous peeking, the
    cumulative Type I error stays bounded by alpha.

    We verify this by simulating many experiments under H0 (no effect),
    running mSPRT sequentially with frequent peeking, and counting how
    often we falsely reject.
    """

    def test_type_one_error_bounded_under_peeking(self):
        """
        Run 1000 H0 experiments with peeking every 25 observations.
        Empirical false-positive rate should be at or below alpha (with
        some Monte Carlo slack).

        Theoretical bound from Wald: P(ever reject | H0) <= alpha.
        With 1000 simulations at alpha=0.05, the binomial SE is ~0.007;
        a generous upper bound is alpha + 3*SE ~ 0.07. We use 0.075.
        """
        rng = np.random.default_rng(2025)
        n_sims = 1000
        n_max = 2000
        alpha = 0.05
        rejections = 0

        for _ in range(n_sims):
            p = rng.uniform(0.05, 0.15)  # vary the true rate across sims
            control = rng.binomial(1, p, n_max)
            treatment = rng.binomial(1, p, n_max)  # same rate -> H0
            history = msprt_sequential(
                control, treatment, alpha_signif=alpha, check_every=25
            )
            if history[-1].reject_null:
                rejections += 1

        rate = rejections / n_sims
        assert rate <= 0.075, (
            f"Empirical Type I error rate {rate:.4f} exceeds bound. "
            f"mSPRT should keep this below alpha={alpha} (with MC slack)."
        )
        # Also assert it's not absurdly conservative (would indicate a bug
        # where we're never rejecting under reasonable conditions)
        # No lower bound check here — Wald is one-sided.


# ============================================================
# CATEGORY 4: VALIDATION
# ============================================================

class TestValidation:
    def test_negative_successes_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            msprt_beta_binomial(-1, 100, 50, 100)

    def test_successes_exceed_total_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            msprt_beta_binomial(150, 100, 50, 100)

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="positive"):
            msprt_beta_binomial(0, 0, 50, 100)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha_signif"):
            msprt_beta_binomial(50, 100, 50, 100, alpha_signif=1.5)
        with pytest.raises(ValueError, match="alpha_signif"):
            msprt_beta_binomial(50, 100, 50, 100, alpha_signif=0.0)

    def test_invalid_prior_raises(self):
        with pytest.raises(ValueError, match="Prior"):
            msprt_beta_binomial(50, 100, 50, 100, prior_alpha=-1.0)