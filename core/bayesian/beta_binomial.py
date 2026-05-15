
"""
Bayesian A/B testing with Beta-Binomial conjugate model.

For two-arm tests with binary outcomes (conversion rates), the Beta
distribution is the conjugate prior to the Binomial likelihood. This
means the posterior is also Beta and the update is closed-form:

    prior:     p ~ Beta(alpha, beta)
    data:      s successes out of n trials
    posterior: p ~ Beta(alpha + s, beta + n - s)

No MCMC needed. We compute "P(B beats A)" via Monte Carlo sampling
from the two posterior distributions — fast, accurate, and easy to
explain at a whiteboard.

Why this matters: Bayesian A/B tests answer the question PMs actually
ask ("what's the chance B is better?") rather than the question
frequentist tests answer ("how surprising is this data under the null?").
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats


@dataclass
class BayesianABResult:
    """Posterior summary for a two-arm Bayesian A/B test."""
    # Posterior parameters for each arm
    alpha_a: float
    beta_a: float
    alpha_b: float
    beta_b: float

    # Posterior means (point estimates of conversion rates)
    posterior_mean_a: float
    posterior_mean_b: float

    # 95% credible intervals for each arm
    credible_interval_a: Tuple[float, float]
    credible_interval_b: Tuple[float, float]

    # The headline number: probability B is better than A
    probability_b_beats_a: float

    # Posterior expected lift (B - A) and its credible interval
    expected_lift: float
    lift_credible_interval: Tuple[float, float]

    # Expected loss if you pick B (when actually A is better) — useful
    # for decision-theoretic stopping rules
    expected_loss_choosing_b: float

    n_mc_samples: int

    def __str__(self) -> str:
        return (
            f"Bayesian A/B Test (Beta-Binomial)\n"
            f"  P(B > A): {self.probability_b_beats_a:.4f}\n"
            f"  Posterior mean A: {self.posterior_mean_a:.4f} "
            f"[{self.credible_interval_a[0]:.4f}, {self.credible_interval_a[1]:.4f}]\n"
            f"  Posterior mean B: {self.posterior_mean_b:.4f} "
            f"[{self.credible_interval_b[0]:.4f}, {self.credible_interval_b[1]:.4f}]\n"
            f"  Expected lift (B - A): {self.expected_lift:+.4f} "
            f"[{self.lift_credible_interval[0]:+.4f}, {self.lift_credible_interval[1]:+.4f}]\n"
            f"  Expected loss of choosing B: {self.expected_loss_choosing_b:.6f}"
        )


def bayesian_ab_test_beta_binomial(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_mc_samples: int = 100_000,
    credible_level: float = 0.95,
    random_state: int = 42,
) -> BayesianABResult:
    """
    Run a Bayesian A/B test on two binary-outcome arms.

    Parameters
    ----------
    control_successes : int
        Conversions in the control arm.
    control_total : int
        Total trials in the control arm.
    treatment_successes : int
        Conversions in the treatment arm.
    treatment_total : int
        Total trials in the treatment arm.
    prior_alpha : float, default 1.0
        Alpha parameter of Beta prior (shared by both arms).
        Defaults to 1.0 for a uniform "I have no prior info" prior.
    prior_beta : float, default 1.0
        Beta parameter of Beta prior. Defaults to 1.0.
    n_mc_samples : int, default 100_000
        Number of Monte Carlo samples for posterior comparison.
        100k gives standard error of ~0.0016 on P(B>A) — plenty precise.
    credible_level : float, default 0.95
        Credible interval level (analogous to frequentist confidence level).
    random_state : int
        For reproducibility of the MC sampling.

    Returns
    -------
    BayesianABResult
    """
    # Input validation
    if control_total <= 0 or treatment_total <= 0:
        raise ValueError("Sample sizes must be positive")
    if control_successes < 0 or treatment_successes < 0:
        raise ValueError("Successes must be non-negative")
    if control_successes > control_total or treatment_successes > treatment_total:
        raise ValueError("Successes cannot exceed total trials")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Beta prior parameters must be positive")
    if n_mc_samples < 1000:
        raise ValueError("Use at least 1000 MC samples for stable estimates")

    # ----------------------------------------------------------
    # STEP 1: Conjugate update — the closed-form posterior
    # ----------------------------------------------------------
    # Posterior_A = Beta(alpha + s_A, beta + n_A - s_A)
    # Posterior_B = Beta(alpha + s_B, beta + n_B - s_B)
    alpha_a = prior_alpha + control_successes
    beta_a = prior_beta + (control_total - control_successes)
    alpha_b = prior_alpha + treatment_successes
    beta_b = prior_beta + (treatment_total - treatment_successes)

    # ----------------------------------------------------------
    # STEP 2: Posterior point estimates (means of each Beta)
    # ----------------------------------------------------------
    # Mean of Beta(alpha, beta) = alpha / (alpha + beta)
    posterior_mean_a = alpha_a / (alpha_a + beta_a)
    posterior_mean_b = alpha_b / (alpha_b + beta_b)

    # ----------------------------------------------------------
    # STEP 3: Credible intervals from the Beta quantile function
    # ----------------------------------------------------------
    lo_q = (1 - credible_level) / 2
    hi_q = 1 - lo_q
    ci_a = (
        stats.beta.ppf(lo_q, alpha_a, beta_a),
        stats.beta.ppf(hi_q, alpha_a, beta_a),
    )
    ci_b = (
        stats.beta.ppf(lo_q, alpha_b, beta_b),
        stats.beta.ppf(hi_q, alpha_b, beta_b),
    )

    # ----------------------------------------------------------
    # STEP 4: Monte Carlo comparison — the core Bayesian quantity
    # ----------------------------------------------------------
    # Draw samples from the two posteriors, compare element-wise.
    # P(B > A) = fraction of samples where the B-draw exceeds the A-draw.
    rng = np.random.default_rng(random_state)
    samples_a = rng.beta(alpha_a, beta_a, size=n_mc_samples)
    samples_b = rng.beta(alpha_b, beta_b, size=n_mc_samples)
    prob_b_beats_a = float(np.mean(samples_b > samples_a))

    # ----------------------------------------------------------
    # STEP 5: Lift distribution (B - A)
    # ----------------------------------------------------------
    lift_samples = samples_b - samples_a
    expected_lift = float(np.mean(lift_samples))
    lift_ci = (
        float(np.quantile(lift_samples, lo_q)),
        float(np.quantile(lift_samples, hi_q)),
    )

    # ----------------------------------------------------------
    # STEP 6: Expected loss of choosing B
    # ----------------------------------------------------------
    # If A is actually better, your loss from choosing B is (A - B).
    # If B is actually better, no loss.
    # Expected loss of B = E[max(A - B, 0)] over the joint posterior.
    # Used in Bayesian stopping rules: stop when expected loss is below
    # some tolerance.
    expected_loss_b = float(np.mean(np.maximum(samples_a - samples_b, 0)))

    return BayesianABResult(
        alpha_a=alpha_a,
        beta_a=beta_a,
        alpha_b=alpha_b,
        beta_b=beta_b,
        posterior_mean_a=posterior_mean_a,
        posterior_mean_b=posterior_mean_b,
        credible_interval_a=ci_a,
        credible_interval_b=ci_b,
        probability_b_beats_a=prob_b_beats_a,
        expected_lift=expected_lift,
        lift_credible_interval=lift_ci,
        expected_loss_choosing_b=expected_loss_b,
        n_mc_samples=n_mc_samples,
    )


if __name__ == "__main__":
    # Demo: small experiment where B looks slightly better
    print("=" * 60)
    print("Bayesian A/B test demo")
    print("=" * 60)
    result = bayesian_ab_test_beta_binomial(
        control_successes=120, control_total=2000,
        treatment_successes=145, treatment_total=2000,
    )
    print(result)