"""
Two-sample statistical tests for A/B testing.

Each test returns a TestResult with all relevant information for downstream analysis.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Standard result format for all two-sample tests."""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    effect_size_name: str
    n_control: int
    n_treatment: int
    interpretation: str

    def __str__(self) -> str:
        return (
            f"{self.test_name}\n"
            f"  Statistic: {self.statistic:.4f}\n"
            f"  P-value: {self.p_value:.4f}\n"
            f"  {self.effect_size_name}: {self.effect_size:.4f}\n"
            f"  95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]\n"
            f"  Sample sizes: control={self.n_control}, treatment={self.n_treatment}\n"
            f"  {self.interpretation}"
        )


def welch_t_test(
    control: np.ndarray, 
    treatment: np.ndarray,
    alpha: float = 0.05
) -> TestResult:
    """
    Welch's t-test for two samples with unequal variances.
    
    Use when: continuous outcomes, no assumption of equal variance.
    Standard for most A/B tests on continuous metrics.
    """
    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
    
    mean_diff = treatment.mean() - control.mean()
    se_diff = np.sqrt(
        treatment.var(ddof=1) / len(treatment) + 
        control.var(ddof=1) / len(control)
    )
    
    df = (treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))**2 / (
        (treatment.var(ddof=1) / len(treatment))**2 / (len(treatment) - 1) +
        (control.var(ddof=1) / len(control))**2 / (len(control) - 1)
    )
    
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    pooled_std = np.sqrt((control.var(ddof=1) + treatment.var(ddof=1)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    if p_val < alpha:
        interpretation = f"Statistically significant difference detected (p={p_val:.4f} < {alpha})"
    else:
        interpretation = f"No significant difference detected (p={p_val:.4f} >= {alpha})"
    
    return TestResult(
        test_name="Welch's t-test",
        statistic=t_stat,
        p_value=p_val,
        confidence_interval=(ci_lower, ci_upper),
        effect_size=cohens_d,
        effect_size_name="Cohen's d",
        n_control=len(control),
        n_treatment=len(treatment),
        interpretation=interpretation
    )


def two_proportion_test(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    alpha: float = 0.05
) -> TestResult:
    """
    Two-proportion z-test for binary outcomes.
    
    Use when: binary outcomes (conversion rate, click-through rate).
    Standard for most A/B tests on conversion metrics.
    """
    p_control = control_successes / control_total
    p_treatment = treatment_successes / treatment_total
    
    p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)
    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
    
    diff = p_treatment - p_control
    z_stat = diff / se_pooled if se_pooled > 0 else 0
    
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    se_unpooled = np.sqrt(
        p_control * (1 - p_control) / control_total + 
        p_treatment * (1 - p_treatment) / treatment_total
    )
    z_critical = stats.norm.ppf(1 - alpha/2)
    ci_lower = diff - z_critical * se_unpooled
    ci_upper = diff + z_critical * se_unpooled
    
    relative_lift = (p_treatment - p_control) / p_control if p_control > 0 else 0
    
    if p_val < alpha:
        interpretation = (
            f"Statistically significant difference detected (p={p_val:.4f} < {alpha}). "
            f"Treatment lift: {relative_lift*100:.2f}%"
        )
    else:
        interpretation = f"No significant difference detected (p={p_val:.4f} >= {alpha})"
    
    return TestResult(
        test_name="Two-proportion z-test",
        statistic=z_stat,
        p_value=p_val,
        confidence_interval=(ci_lower, ci_upper),
        effect_size=relative_lift,
        effect_size_name="Relative lift",
        n_control=control_total,
        n_treatment=treatment_total,
        interpretation=interpretation
    )


if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("Test 1: Continuous outcome (Welch's t-test)")
    print("=" * 60)
    control = np.random.normal(10, 2, 1000)
    treatment = np.random.normal(10.5, 2, 1000)
    result1 = welch_t_test(control, treatment)
    print(result1)
    print()
    
    print("=" * 60)
    print("Test 2: Binary outcome (Two-proportion z-test)")
    print("=" * 60)
    result2 = two_proportion_test(
        control_successes=50, control_total=1000,
        treatment_successes=70, treatment_total=1000
    )
    print(result2)
