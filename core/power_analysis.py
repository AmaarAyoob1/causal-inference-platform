"""
Power analysis for A/B testing.

Provides sample size calculation, minimum detectable effect (MDE)
estimation, and post-hoc power computation for two common designs:
- Two-proportion z-test (binary outcomes)
- Two-sample t-test (continuous outcomes, equal variances assumed)

Design philosophy:
- Each function answers ONE of the three power-analysis questions
  (solve for n, solve for effect, solve for power).
- We wrap statsmodels for correctness, but expose a clean API with
  validation and a structured result type.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass
class PowerAnalysisResult:
    """Structured result for any power analysis computation."""
    test_type: str
    sample_size_per_group: int
    effect_size: float
    effect_size_name: str
    alpha: float
    power: float
    notes: str

    def __str__(self) -> str:
        return (
            f"{self.test_type} Power Analysis\n"
            f"  Sample size per group: {self.sample_size_per_group:,}\n"
            f"  {self.effect_size_name}: {self.effect_size:.4f}\n"
            f"  Alpha: {self.alpha}\n"
            f"  Power: {self.power:.3f}\n"
            f"  {self.notes}"
        )


# ============================================================
# TWO-PROPORTION POWER ANALYSIS
# ============================================================

def sample_size_for_proportions(
    baseline_rate: float,
    minimum_detectable_lift: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> PowerAnalysisResult:
    """
    Compute required sample size per group for a two-proportion z-test.

    Parameters
    ----------
    baseline_rate : float in (0, 1)
        Expected conversion rate of the control group.
    minimum_detectable_lift : float
        The relative lift we want to be able to detect, expressed as a
        proportion (NOT a percentage). E.g., 0.02 means "detect a 2%
        relative lift over baseline" -> treatment rate = baseline * 1.02.

        Why relative lift instead of absolute? Because in practice, PMs
        and stakeholders specify effects relative to baseline ("we want
        to see at least a 5% lift in conversion"). The math underneath
        is in absolute terms; this function converts for you.
    alpha : float, default 0.05
        Significance level (Type I error rate).
    power : float, default 0.80
        Desired power (1 - Type II error rate).
    alternative : {"two-sided", "larger", "smaller"}
        Direction of the test.

    Returns
    -------
    PowerAnalysisResult
    """
    # Input validation
    if not (0 < baseline_rate < 1):
        raise ValueError(
            f"baseline_rate must be in (0, 1), got {baseline_rate}"
        )
    if minimum_detectable_lift <= 0:
        raise ValueError(
            f"minimum_detectable_lift must be > 0, got {minimum_detectable_lift}"
        )
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0 < power < 1):
        raise ValueError(f"power must be in (0, 1), got {power}")

    treatment_rate = baseline_rate * (1 + minimum_detectable_lift)
    if treatment_rate >= 1:
        raise ValueError(
            f"baseline * (1 + lift) = {treatment_rate:.3f} exceeds 1.0. "
            f"Lift too large for this baseline (conversion rates can't "
            f"exceed 100%)."
        )

    # Cohen's h is the standard effect size for comparing proportions.
    # It's the arcsine-transformed difference; the transformation
    # stabilizes variance across the (0, 1) range.
    effect_size = proportion_effectsize(treatment_rate, baseline_rate)

    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        alternative=alternative,
        ratio=1.0,  # equal-sized groups
    )

    return PowerAnalysisResult(
        test_type="Two-proportion z-test",
        sample_size_per_group=int(np.ceil(n)),
        effect_size=effect_size,
        effect_size_name="Cohen's h",
        alpha=alpha,
        power=power,
        notes=(
            f"Baseline rate: {baseline_rate:.4f}, "
            f"Treatment rate: {treatment_rate:.4f} "
            f"({minimum_detectable_lift*100:+.2f}% relative lift)"
        ),
    )


def mde_for_proportions(
    baseline_rate: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> PowerAnalysisResult:
    """
    Solve for the minimum detectable effect given fixed sample size.

    Inverts the question: "I have N users per arm. What's the smallest
    relative lift I can detect with the specified power?"

    Returned `effect_size` is in Cohen's h units; `notes` reports the
    implied relative lift over baseline.
    """
    if not (0 < baseline_rate < 1):
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    if sample_size_per_group < 2:
        raise ValueError(
            f"sample_size_per_group must be >= 2, got {sample_size_per_group}"
        )

    analysis = NormalIndPower()
    h_mde = analysis.solve_power(
        nobs1=sample_size_per_group,
        alpha=alpha,
        power=power,
        alternative=alternative,
        ratio=1.0,
    )

    # Invert Cohen's h to find the treatment rate that produces this h.
    # h = 2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1)))
    # => p2 = sin(arcsin(sqrt(p1)) + h/2)^2
    phi1 = np.arcsin(np.sqrt(baseline_rate))
    phi2 = phi1 + h_mde / 2
    treatment_rate = np.sin(phi2) ** 2
    relative_lift = (treatment_rate - baseline_rate) / baseline_rate

    return PowerAnalysisResult(
        test_type="Two-proportion z-test (MDE)",
        sample_size_per_group=sample_size_per_group,
        effect_size=h_mde,
        effect_size_name="Cohen's h (MDE)",
        alpha=alpha,
        power=power,
        notes=(
            f"Baseline: {baseline_rate:.4f}, "
            f"MDE treatment rate: {treatment_rate:.4f} "
            f"({relative_lift*100:+.2f}% min detectable lift)"
        ),
    )


# ============================================================
# CONTINUOUS-OUTCOME POWER ANALYSIS (T-TEST)
# ============================================================

def sample_size_for_means(
    cohens_d: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> PowerAnalysisResult:
    """
    Compute required sample size per group for a two-sample t-test.

    Parameters
    ----------
    cohens_d : float
        Standardized effect size: d = (mu_2 - mu_1) / sigma_pooled.
        Conventional benchmarks: 0.2 = small, 0.5 = medium, 0.8 = large.
        Negative values are accepted; only magnitude matters for n.
    """
    if cohens_d == 0:
        raise ValueError(
            "Cohen's d cannot be zero. With no effect, no finite sample "
            "size achieves nonzero power above alpha."
        )

    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=abs(cohens_d),
        alpha=alpha,
        power=power,
        alternative=alternative,
        ratio=1.0,
    )

    return PowerAnalysisResult(
        test_type="Two-sample t-test",
        sample_size_per_group=int(np.ceil(n)),
        effect_size=cohens_d,
        effect_size_name="Cohen's d",
        alpha=alpha,
        power=power,
        notes="Assumes equal variances across groups",
    )


def power_post_hoc(
    cohens_d: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> PowerAnalysisResult:
    """
    Compute achieved power given fixed effect size and sample size.

    Important caveat: post-hoc power (computed on an OBSERVED effect
    after the experiment) is generally a bad idea — it's a deterministic
    function of the p-value and adds no information. This function is
    intended for use BEFORE running an experiment (sensitivity analysis),
    not after.
    """
    if cohens_d == 0:
        raise ValueError("Cohen's d cannot be zero")
    if sample_size_per_group < 2:
        raise ValueError(
            f"sample_size_per_group must be >= 2, got {sample_size_per_group}"
        )

    analysis = TTestIndPower()
    power = analysis.solve_power(
        effect_size=abs(cohens_d),
        nobs1=sample_size_per_group,
        alpha=alpha,
        ratio=1.0,
        alternative=alternative,
        power=None,
    )

    return PowerAnalysisResult(
        test_type="Two-sample t-test (post-hoc power)",
        sample_size_per_group=sample_size_per_group,
        effect_size=cohens_d,
        effect_size_name="Cohen's d",
        alpha=alpha,
        power=float(power),
        notes="Sensitivity analysis. Avoid using on observed effects.",
    )


if __name__ == "__main__":
    # Demo: how big a sample do we need for a typical e-commerce A/B test?
    print("=" * 60)
    print("Power analysis demo: e-commerce conversion test")
    print("=" * 60)

    result = sample_size_for_proportions(
        baseline_rate=0.05,        # 5% baseline conversion
        minimum_detectable_lift=0.05,  # detect a 5% relative lift -> 5.25%
        alpha=0.05,
        power=0.80,
    )
    print(result)
    print()

    print("=" * 60)
    print("Inverse: 'I have 10,000 users per arm. What's my MDE?'")
    print("=" * 60)
    result = mde_for_proportions(
        baseline_rate=0.05,
        sample_size_per_group=10_000,
    )
    print(result)