"""
Bayesian A/B Test page.

Beta-Binomial conjugate model for two-arm binary outcome experiments.
Visualizes posteriors, lift distribution, and expected loss alongside
the headline P(B > A) probability.

Design priorities:
- PM-friendly headline metric at the top
- Posterior overlap plot as the centerpiece visualization
- Lift distribution histogram showing where probability mass sits
- Decision-theoretic numbers (expected loss) for stopping-rule context
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from core.bayesian.beta_binomial import bayesian_ab_test_beta_binomial


st.set_page_config(page_title="Bayesian A/B Test", page_icon="🎲", layout="wide")
st.title("🎲 Bayesian A/B Test")
st.markdown(
    "Compute the posterior probability that B beats A using a Beta-Binomial "
    "conjugate model. Unlike a p-value, this directly answers the question "
    "PMs actually ask: *'what's the chance the treatment is better?'*"
)


# ============================================================
# SIDEBAR: priors and educational context
# ============================================================
with st.sidebar:
    st.header("Prior settings")
    st.caption(
        "The Beta(α, β) prior encodes your belief about the conversion "
        "rate before seeing data. Beta(1, 1) is uniform — agnostic. "
        "Heavier priors require more data to override."
    )

    prior_alpha = st.number_input(
        "Prior α (pseudo-successes)",
        min_value=0.1,
        max_value=1000.0,
        value=1.0,
        step=0.5,
        help=(
            "Think of α as 'fake successes you saw before the experiment.' "
            "Higher α + β = stronger prior."
        ),
    )
    prior_beta = st.number_input(
        "Prior β (pseudo-failures)",
        min_value=0.1,
        max_value=1000.0,
        value=1.0,
        step=0.5,
    )

    prior_mean = prior_alpha / (prior_alpha + prior_beta)
    prior_strength = prior_alpha + prior_beta
    st.caption(
        f"Prior mean: {prior_mean:.3f} "
        f"(equivalent to {prior_strength:.0f} fake observations)"
    )

    st.divider()
    st.header("Computation settings")
    n_mc_samples = st.select_slider(
        "Monte Carlo samples",
        options=[10_000, 50_000, 100_000, 500_000],
        value=100_000,
        help=(
            "More samples = more precise P(B>A) estimate. "
            "100k gives standard error ~0.0016 on the probability."
        ),
    )


# ============================================================
# MAIN: data inputs
# ============================================================
st.subheader("Experiment data")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Control (A)**")
    control_total = st.number_input(
        "Total trials (A)",
        min_value=1,
        max_value=10_000_000,
        value=2000,
        step=100,
        key="ctrl_n",
    )
    control_successes = st.number_input(
        "Successes (A)",
        min_value=0,
        max_value=int(control_total),
        value=min(120, int(control_total)),
        step=10,
        key="ctrl_s",
    )

with col2:
    st.markdown("**Treatment (B)**")
    treatment_total = st.number_input(
        "Total trials (B)",
        min_value=1,
        max_value=10_000_000,
        value=2000,
        step=100,
        key="trt_n",
    )
    treatment_successes = st.number_input(
        "Successes (B)",
        min_value=0,
        max_value=int(treatment_total),
        value=min(145, int(treatment_total)),
        step=10,
        key="trt_s",
    )

empirical_a = control_successes / control_total if control_total > 0 else 0
empirical_b = treatment_successes / treatment_total if treatment_total > 0 else 0
st.caption(
    f"Empirical rates: A = {empirical_a:.4f} ({control_successes}/{control_total}), "
    f"B = {empirical_b:.4f} ({treatment_successes}/{treatment_total})"
)

st.divider()


# ============================================================
# RUN THE TEST
# ============================================================
try:
    result = bayesian_ab_test_beta_binomial(
        control_successes=control_successes,
        control_total=control_total,
        treatment_successes=treatment_successes,
        treatment_total=treatment_total,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        n_mc_samples=n_mc_samples,
    )

    # ----------------------------------------------------------
    # HEADLINE
    # ----------------------------------------------------------
    prob = result.probability_b_beats_a
    if prob >= 0.95:
        verdict = "Strong evidence B is better."
        verdict_color = "success"
    elif prob >= 0.80:
        verdict = "Moderate evidence B is better."
        verdict_color = "info"
    elif prob <= 0.05:
        verdict = "Strong evidence A is better."
        verdict_color = "error"
    elif prob <= 0.20:
        verdict = "Moderate evidence A is better."
        verdict_color = "warning"
    else:
        verdict = "Inconclusive — continue the experiment or accept the uncertainty."
        verdict_color = "warning"

    h1, h2, h3 = st.columns([2, 1, 1])
    with h1:
        st.metric(
            "P(B > A)",
            f"{prob:.2%}",
            help=(
                "Posterior probability that B's true conversion rate "
                "exceeds A's, given the observed data."
            ),
        )
    with h2:
        st.metric(
            "Expected lift (B − A)",
            f"{result.expected_lift:+.4f}",
            delta=f"{(result.expected_lift / empirical_a * 100):+.1f}% rel" if empirical_a > 0 else None,
        )
    with h3:
        st.metric(
            "Expected loss of choosing B",
            f"{result.expected_loss_choosing_b:.5f}",
            help=(
                "If you ship B but A is actually better, your loss is "
                "(A − B). This is the posterior expected loss — used in "
                "Bayesian stopping rules."
            ),
        )

    getattr(st, verdict_color)(verdict)

    st.divider()

    # ----------------------------------------------------------
    # POSTERIOR OVERLAP PLOT — the centerpiece
    # ----------------------------------------------------------
    st.subheader("Posterior distributions")
    st.caption(
        "Two Beta posteriors. The further apart and the less overlap, the "
        "more confident we can be that B differs from A."
    )

    # Pick an x-axis range that spans both posteriors comfortably
    x_min = max(
        0.0,
        min(
            stats.beta.ppf(0.001, result.alpha_a, result.beta_a),
            stats.beta.ppf(0.001, result.alpha_b, result.beta_b),
        ),
    )
    x_max = min(
        1.0,
        max(
            stats.beta.ppf(0.999, result.alpha_a, result.beta_a),
            stats.beta.ppf(0.999, result.alpha_b, result.beta_b),
        ),
    )
    x = np.linspace(x_min, x_max, 500)
    pdf_a = stats.beta.pdf(x, result.alpha_a, result.beta_a)
    pdf_b = stats.beta.pdf(x, result.alpha_b, result.beta_b)

    fig_posterior = go.Figure()
    fig_posterior.add_trace(
        go.Scatter(
            x=x, y=pdf_a,
            name=f"Control A: Beta({result.alpha_a:.0f}, {result.beta_a:.0f})",
            fill="tozeroy",
            line=dict(width=2),
            opacity=0.6,
        )
    )
    fig_posterior.add_trace(
        go.Scatter(
            x=x, y=pdf_b,
            name=f"Treatment B: Beta({result.alpha_b:.0f}, {result.beta_b:.0f})",
            fill="tozeroy",
            line=dict(width=2),
            opacity=0.6,
        )
    )
    fig_posterior.add_vline(
        x=result.posterior_mean_a,
        line=dict(dash="dash"),
        annotation_text=f"A mean: {result.posterior_mean_a:.4f}",
        annotation_position="top left",
    )
    fig_posterior.add_vline(
        x=result.posterior_mean_b,
        line=dict(dash="dash"),
        annotation_text=f"B mean: {result.posterior_mean_b:.4f}",
        annotation_position="top right",
    )
    fig_posterior.update_layout(
        xaxis_title="Conversion rate",
        yaxis_title="Posterior density",
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_posterior, use_container_width=True)

    # ----------------------------------------------------------
    # LIFT DISTRIBUTION
    # ----------------------------------------------------------
    st.subheader("Posterior distribution of the lift (B − A)")
    st.caption(
        "Sampled differences between B and A draws from their posteriors. "
        "P(B > A) is the fraction of mass to the right of zero."
    )

    # Re-sample for the histogram (small enough not to be slow)
    rng = np.random.default_rng(42)
    samples_a = rng.beta(result.alpha_a, result.beta_a, size=50_000)
    samples_b = rng.beta(result.alpha_b, result.beta_b, size=50_000)
    lift_samples = samples_b - samples_a

    fig_lift = go.Figure()
    fig_lift.add_trace(
        go.Histogram(
            x=lift_samples,
            nbinsx=80,
            histnorm="probability density",
            name="Lift (B − A)",
            opacity=0.75,
        )
    )
    fig_lift.add_vline(
        x=0,
        line=dict(color="red", width=2),
        annotation_text="No effect",
        annotation_position="top",
    )
    fig_lift.add_vline(
        x=result.expected_lift,
        line=dict(dash="dash"),
        annotation_text=f"Mean: {result.expected_lift:+.4f}",
        annotation_position="bottom",
    )
    fig_lift.update_layout(
        xaxis_title="Lift (B − A)",
        yaxis_title="Posterior density",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_lift, use_container_width=True)

    pct_above_zero = float(np.mean(lift_samples > 0)) * 100
    st.caption(
        f"{pct_above_zero:.1f}% of the posterior mass is above zero "
        f"(this is what P(B > A) measures). "
        f"95% credible interval for the lift: "
        f"[{result.lift_credible_interval[0]:+.4f}, "
        f"{result.lift_credible_interval[1]:+.4f}]."
    )

    # ----------------------------------------------------------
    # DETAILED SUMMARY (for analysts who want more)
    # ----------------------------------------------------------
    with st.expander("Detailed posterior summary"):
        st.write(
            f"""
            **Posterior A:** Beta({result.alpha_a:.0f}, {result.beta_a:.0f})
            - Mean: {result.posterior_mean_a:.4f}
            - 95% credible interval: [{result.credible_interval_a[0]:.4f}, {result.credible_interval_a[1]:.4f}]

            **Posterior B:** Beta({result.alpha_b:.0f}, {result.beta_b:.0f})
            - Mean: {result.posterior_mean_b:.4f}
            - 95% credible interval: [{result.credible_interval_b[0]:.4f}, {result.credible_interval_b[1]:.4f}]

            **Lift (B − A):**
            - Expected value: {result.expected_lift:+.4f}
            - 95% credible interval: [{result.lift_credible_interval[0]:+.4f}, {result.lift_credible_interval[1]:+.4f}]

            **Decision metrics:**
            - P(B > A): {result.probability_b_beats_a:.4f}
            - Expected loss of choosing B: {result.expected_loss_choosing_b:.6f}
            - Monte Carlo samples used: {result.n_mc_samples:,}
            """
        )

except ValueError as e:
    st.error(f"Invalid input: {e}")