"""
Mixture Sequential Probability Ratio Test (mSPRT) for two-arm
Beta-Binomial A/B testing.

Peeking-safe alternative to fixed-horizon frequentist tests.
You can monitor the likelihood ratio continuously and stop as soon as
it crosses the rejection boundary — the cumulative Type I error is
bounded by alpha *regardless of how often you peek*.

This is what production experimentation platforms (Optimizely Stats
Engine, parts of what's used at Meta/Airbnb/Etsy) actually run.

Math sketch
-----------
For two arms with Beta-Binomial likelihood and a Beta(alpha, beta)
mixing prior on the conversion rates, the mSPRT likelihood ratio
after observing (s_A, n_A) and (s_B, n_B) reduces to a closed form
in terms of Beta functions B(.,.):

    Lambda_n = [B(alpha+s_A, beta+n_A-s_A) * B(alpha+s_B, beta+n_B-s_B)
                / B(alpha, beta)^2]
                  /
               [B(2*alpha+s_A+s_B, 2*beta+n_A+n_B-s_A-s_B) / B(2*alpha, 2*beta)]

Reject H_0 ("no difference") when Lambda_n >= 1/alpha_signif.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.special import betaln  # log-Beta function for numerical stability


@dataclass
class mSPRTResult:
    """Result of evaluating mSPRT at a single point in time."""
    log_likelihood_ratio: float
    likelihood_ratio: float
    rejection_threshold: float  # 1 / alpha_signif (on the LR scale)
    log_rejection_threshold: float  # log(1 / alpha_signif)
    reject_null: bool
    n_a: int
    n_b: int
    s_a: int
    s_b: int
    p_hat_a: float
    p_hat_b: float

    def __str__(self) -> str:
        decision = "REJECT H0 (B differs from A)" if self.reject_null else "Continue / fail to reject"
        return (
            f"mSPRT @ n_A={self.n_a}, n_B={self.n_b}\n"
            f"  p_hat_A = {self.p_hat_a:.4f}  ({self.s_a}/{self.n_a})\n"
            f"  p_hat_B = {self.p_hat_b:.4f}  ({self.s_b}/{self.n_b})\n"
            f"  log Lambda = {self.log_likelihood_ratio:+.3f}\n"
            f"  Lambda     = {self.likelihood_ratio:.4f}\n"
            f"  Threshold  = {self.rejection_threshold:.4f} "
            f"(log = {self.log_rejection_threshold:.3f})\n"
            f"  Decision   = {decision}"
        )


def msprt_beta_binomial(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    alpha_signif: float = 0.05,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> mSPRTResult:
    """
    Compute the mSPRT log-likelihood ratio at a single point in time
    and decide whether to reject the null of "no difference between arms."

    Use log-space throughout for numerical stability — likelihood ratios
    explode quickly and overflow float64 in raw form.

    Parameters
    ----------
    control_successes, control_total : int
        Observed successes and trials for the control arm so far.
    treatment_successes, treatment_total : int
        Observed successes and trials for the treatment arm so far.
    alpha_signif : float
        Desired Type I error rate. Rejection threshold is 1/alpha_signif.
    prior_alpha, prior_beta : float
        Mixing prior parameters. Beta(1,1) = uniform, the conventional
        default. Stronger priors require more evidence to reject.
    """
    if control_total <= 0 or treatment_total <= 0:
        raise ValueError("Sample sizes must be positive")
    if control_successes < 0 or treatment_successes < 0:
        raise ValueError("Successes must be non-negative")
    if control_successes > control_total or treatment_successes > treatment_total:
        raise ValueError("Successes cannot exceed total trials")
    if not (0 < alpha_signif < 1):
        raise ValueError("alpha_signif must be in (0, 1)")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Prior parameters must be positive")

    s_a, n_a = control_successes, control_total
    s_b, n_b = treatment_successes, treatment_total
    a, b = prior_alpha, prior_beta

    # All computation in log-space.
    # log Lambda_n
    #   = log [B(a+s_A, b+n_A-s_A) * B(a+s_B, b+n_B-s_B)]
    #   - 2 * log B(a, b)
    #   - log B(2a + s_A + s_B, 2b + n_A + n_B - s_A - s_B)
    #   + log B(2a, 2b)
    log_num = (
        betaln(a + s_a, b + n_a - s_a)
        + betaln(a + s_b, b + n_b - s_b)
        - 2 * betaln(a, b)
    )
    log_den = (
        betaln(2 * a + s_a + s_b, 2 * b + (n_a - s_a) + (n_b - s_b))
        - betaln(2 * a, 2 * b)
    )
    log_lambda = log_num - log_den

    log_threshold = -np.log(alpha_signif)  # log(1 / alpha_signif)
    reject = log_lambda >= log_threshold

    # Convert to actual lambda only for display; computation stays in log.
    # Clip to avoid overflow when log_lambda is huge.
    lambda_val = float(np.exp(min(log_lambda, 700.0)))

    return mSPRTResult(
        log_likelihood_ratio=float(log_lambda),
        likelihood_ratio=lambda_val,
        rejection_threshold=float(np.exp(log_threshold)),
        log_rejection_threshold=float(log_threshold),
        reject_null=bool(reject),
        n_a=n_a,
        n_b=n_b,
        s_a=s_a,
        s_b=s_b,
        p_hat_a=s_a / n_a,
        p_hat_b=s_b / n_b,
    )


def msprt_sequential(
    control_outcomes: np.ndarray,
    treatment_outcomes: np.ndarray,
    alpha_signif: float = 0.05,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    check_every: int = 1,
) -> List[mSPRTResult]:
    """
    Run mSPRT sequentially over a streaming dataset.

    Walks through the outcome arrays one observation at a time (or every
    `check_every` observations) and returns the mSPRT decision at each
    checkpoint. Stops returning further checkpoints once H0 is rejected.

    Parameters
    ----------
    control_outcomes, treatment_outcomes : 1D arrays of 0/1
        Sequential per-user outcomes in the order they arrived.
    check_every : int
        How many observations between LR evaluations. 1 = check every
        observation. Larger values = less granular but faster simulations.
    """
    control_outcomes = np.asarray(control_outcomes, dtype=int)
    treatment_outcomes = np.asarray(treatment_outcomes, dtype=int)

    n_max = min(len(control_outcomes), len(treatment_outcomes))
    results: List[mSPRTResult] = []

    for n in range(check_every, n_max + 1, check_every):
        s_a = int(control_outcomes[:n].sum())
        s_b = int(treatment_outcomes[:n].sum())
        result = msprt_beta_binomial(
            control_successes=s_a,
            control_total=n,
            treatment_successes=s_b,
            treatment_total=n,
            alpha_signif=alpha_signif,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )
        results.append(result)
        if result.reject_null:
            break

    return results


if __name__ == "__main__":
    # Demo 1: a "clear winner" experiment
    print("=" * 60)
    print("Demo 1: real lift (6% vs 8%), 5000 users per arm")
    print("=" * 60)
    rng = np.random.default_rng(42)
    control = rng.binomial(1, 0.06, 5000)
    treatment = rng.binomial(1, 0.08, 5000)
    history = msprt_sequential(control, treatment, check_every=50)
    final = history[-1]
    print(final)
    print(f"\nStopped after {final.n_a} observations per arm "
          f"(out of 5000 planned).")
    print(f"Total checks: {len(history)}")

    # Demo 2: no real effect — should NOT reject, even with many peeks
    print("\n" + "=" * 60)
    print("Demo 2: no real lift (6% vs 6%), 5000 users per arm")
    print("=" * 60)
    rng = np.random.default_rng(7)
    control = rng.binomial(1, 0.06, 5000)
    treatment = rng.binomial(1, 0.06, 5000)
    history = msprt_sequential(control, treatment, check_every=50)
    final = history[-1]
    print(final)
    if not final.reject_null:
        print("\nCorrectly DID NOT REJECT despite peeking every 50 obs.")