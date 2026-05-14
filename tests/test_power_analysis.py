"""
Tests for power_analysis module.

Categories:
1. Known-answer tests (sanity checks against published values & statsmodels)
2. Mathematical invariant tests (relationships that MUST hold)
3. Round-trip tests (n_for_effect should invert mde_for_n)
4. Edge case / validation tests
"""
import numpy as np
import pytest

from core.power_analysis import (
    sample_size_for_proportions,
    mde_for_proportions,
    sample_size_for_means,
    power_post_hoc,
)


# ============================================================
# CATEGORY 1: KNOWN-ANSWER / SANITY TESTS
# ============================================================

class TestProportionSampleSize:
    def test_basic_call_returns_valid_result(self):
        result = sample_size_for_proportions(
            baseline_rate=0.10,
            minimum_detectable_lift=0.10,
            alpha=0.05,
            power=0.80,
        )
        assert result.sample_size_per_group > 0
        assert result.power == 0.80
        assert result.alpha == 0.05

    def test_smaller_effect_needs_larger_sample(self):
        """
        Fundamental invariant: holding everything else fixed, smaller
        effects require larger samples. This is the single most important
        property of power analysis.
        """
        small_lift = sample_size_for_proportions(
            baseline_rate=0.10, minimum_detectable_lift=0.05
        )
        large_lift = sample_size_for_proportions(
            baseline_rate=0.10, minimum_detectable_lift=0.20
        )
        assert small_lift.sample_size_per_group > large_lift.sample_size_per_group

    def test_higher_power_needs_larger_sample(self):
        """Holding effect fixed, requiring more power costs more samples."""
        low_power = sample_size_for_proportions(
            baseline_rate=0.10, minimum_detectable_lift=0.10, power=0.70
        )
        high_power = sample_size_for_proportions(
            baseline_rate=0.10, minimum_detectable_lift=0.10, power=0.95
        )
        assert high_power.sample_size_per_group > low_power.sample_size_per_group

    def test_stricter_alpha_needs_larger_sample(self):
        """Smaller alpha (stricter Type I control) requires more samples."""
        loose = sample_size_for_proportions(
            baseline_rate=0.10, minimum_detectable_lift=0.10, alpha=0.10
        )
        strict = sample_size_for_proportions(
            baseline_rate=0.10, minimum_detectable_lift=0.10, alpha=0.01
        )
        assert strict.sample_size_per_group > loose.sample_size_per_group

    def test_quartering_effect_roughly_quadruples_sample(self):
        """
        Sample size scales approximately with 1/effect^2.
        Halving the effect should roughly quadruple n.
        Tolerance is loose because Cohen's h is nonlinear in the raw lift.
        """
        n1 = sample_size_for_proportions(
            baseline_rate=0.10, minimum_detectable_lift=0.10
        ).sample_size_per_group
        n2 = sample_size_for_proportions(
            baseline_rate=0.10, minimum_detectable_lift=0.05
        ).sample_size_per_group
        ratio = n2 / n1
        assert 3.5 < ratio < 4.5, f"Expected ~4x scaling, got {ratio:.2f}x"


# ============================================================
# CATEGORY 2: MDE TESTS
# ============================================================

class TestMDE:
    def test_more_samples_smaller_mde(self):
        """
        Inverse of the sample-size property: more users means we can
        detect smaller effects.
        """
        small_n = mde_for_proportions(baseline_rate=0.10, sample_size_per_group=1000)
        large_n = mde_for_proportions(baseline_rate=0.10, sample_size_per_group=100_000)
        assert abs(large_n.effect_size) < abs(small_n.effect_size)

    def test_mde_inverts_sample_size(self):
        """
        Round-trip property: if I compute n needed for a given lift,
        then compute the MDE at that n, I should recover (approximately)
        the original lift.

        This is a strong correctness check — if either function is wrong,
        this test breaks.
        """
        baseline = 0.10
        target_lift = 0.15

        # Step 1: how many users do I need to detect a 15% lift?
        sized = sample_size_for_proportions(
            baseline_rate=baseline,
            minimum_detectable_lift=target_lift,
            alpha=0.05,
            power=0.80,
        )

        # Step 2: at that n, what's the MDE? Should be close to 15%.
        mde = mde_for_proportions(
            baseline_rate=baseline,
            sample_size_per_group=sized.sample_size_per_group,
            alpha=0.05,
            power=0.80,
        )

        # Extract the implied lift from the MDE result. We parse the notes
        # string here, which is fragile; in production we'd return the
        # lift as a structured field.
        # For robustness, re-derive the lift from the treatment rate:
        # the MDE returns a Cohen's h; convert back to relative lift.
        phi1 = np.arcsin(np.sqrt(baseline))
        phi2 = phi1 + mde.effect_size / 2
        treatment_rate = np.sin(phi2) ** 2
        implied_lift = (treatment_rate - baseline) / baseline

        # We expect implied_lift ~ target_lift. Allow some slack because
        # we round n up to the nearest integer.
        assert abs(implied_lift - target_lift) < 0.005, (
            f"Round-trip failed: target_lift={target_lift}, "
            f"implied_lift={implied_lift}"
        )


# ============================================================
# CATEGORY 3: CONTINUOUS-OUTCOME TESTS
# ============================================================

class TestContinuousPower:
    def test_cohens_d_05_n_around_64(self):
        """
        Well-known textbook result: detecting Cohen's d = 0.5 with
        alpha=0.05 two-sided and 80% power requires n ~ 64 per group.
        This is THE canonical number in stats education for sample sizing.
        """
        result = sample_size_for_means(cohens_d=0.5)
        assert 60 <= result.sample_size_per_group <= 68, (
            f"Expected ~64, got {result.sample_size_per_group}"
        )

    def test_cohens_d_02_much_larger(self):
        """d=0.2 (small effect) should require many times more samples."""
        d05 = sample_size_for_means(cohens_d=0.5).sample_size_per_group
        d02 = sample_size_for_means(cohens_d=0.2).sample_size_per_group
        # Theoretical ratio: (0.5/0.2)^2 = 6.25
        ratio = d02 / d05
        assert 5.5 < ratio < 7.0, f"Expected ~6.25x scaling, got {ratio:.2f}x"

    def test_post_hoc_power_recovers_target(self):
        """
        If I size for 80% power at d=0.5, then compute post-hoc power
        at that exact n and d, I should get ~80% back.
        """
        sized = sample_size_for_means(cohens_d=0.5, power=0.80)
        post = power_post_hoc(
            cohens_d=0.5,
            sample_size_per_group=sized.sample_size_per_group,
        )
        assert 0.79 < post.power < 0.83


# ============================================================
# CATEGORY 4: VALIDATION
# ============================================================

class TestValidation:
    def test_invalid_baseline_rate_raises(self):
        with pytest.raises(ValueError, match="baseline_rate"):
            sample_size_for_proportions(baseline_rate=1.5, minimum_detectable_lift=0.05)
        with pytest.raises(ValueError, match="baseline_rate"):
            sample_size_for_proportions(baseline_rate=0.0, minimum_detectable_lift=0.05)

    def test_invalid_lift_raises(self):
        with pytest.raises(ValueError, match="minimum_detectable_lift"):
            sample_size_for_proportions(baseline_rate=0.10, minimum_detectable_lift=-0.05)

    def test_lift_pushing_past_100_percent_raises(self):
        # Baseline 0.6, lift 100% -> treatment rate 1.2, invalid
        with pytest.raises(ValueError, match="exceeds 1.0"):
            sample_size_for_proportions(baseline_rate=0.6, minimum_detectable_lift=1.0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            sample_size_for_proportions(
                baseline_rate=0.1, minimum_detectable_lift=0.05, alpha=1.5
            )

    def test_zero_cohens_d_raises(self):
        with pytest.raises(ValueError, match="zero"):
            sample_size_for_means(cohens_d=0.0)