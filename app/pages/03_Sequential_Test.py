"""
Sequential Testing page (mSPRT).

Two workflows:
1. Live experiment monitor — feed in cumulative data and see where mSPRT
   stands right now: continue or stop?
2. Operating characteristics simulator — run hypothetical experiments
   under different true effects and visualize the stopping distribution
   and false-positive control.

Design priorities:
- The likelihood ratio trajectory chart is the centerpiece. It tells the
  story of "evidence accumulating over time" in a way numbers don't.
- The OC simulator is the "proof" — visually demonstrating that mSPRT
  does what it claims under H0.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.sequential.msprt import (
    msprt_beta_binomial,
    msprt_sequential,
)


st.set_page_config(
    page_title="Sequential Testing (mSPRT)",
    page_icon="📈",
    layout="wide",
)
st.title("📈 Sequential Testing (mSPRT)")
st.markdown(
    "Peek at your experiment as often as you want without breaking the "
    "statistics. mSPRT (Mixture Sequential Probability Ratio Test) bounds "
    "the cumulative Type I error at α regardless of how many times you "
    "check, so you can stop early when evidence is conclusive."
)


# ============================================================
# SIDEBAR: shared settings
# ============================================================
with st.sidebar:
    st.header("Test settings")
    alpha_signif = st.number_input(
        "Significance level (α)",
        min_value=0.001,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.3f",
        help="Cumulative Type I error budget. mSPRT bounds the probability "
             "of ever rejecting H0 (across all peeks) by this value.",
    )
    prior_alpha = st.number_input(
        "Mixing prior α",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.5,
        help="Beta prior parameter for the conjugate model. 1.0 = uniform.",
    )
    prior_beta = st.number_input(
        "Mixing prior β",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.5,
    )

    st.divider()
    st.caption(
        f"Rejection threshold: Λ ≥ {1/alpha_signif:.1f}  "
        f"(log = {-np.log(alpha_signif):.3f})"
    )


# ============================================================
# TABS
# ============================================================
tab_live, tab_oc = st.tabs(["Live experiment monitor", "Operating characteristics"])


# ------------------------------------------------------------
# TAB 1: LIVE MONITOR
# ------------------------------------------------------------
with tab_live:
    st.subheader("Where does my experiment stand right now?")
    st.markdown(
        "Enter your cumulative experiment data. mSPRT will tell you "
        "whether you have enough evidence to stop, or whether to keep "
        "collecting."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Control (A)**")
        ctrl_n_live = st.number_input(
            "Total trials (A)", min_value=1, value=1500, step=50,
            key="live_ctrl_n",
        )
        ctrl_s_live = st.number_input(
            "Successes (A)", min_value=0, max_value=int(ctrl_n_live),
            value=min(85, int(ctrl_n_live)), step=10, key="live_ctrl_s",
        )

    with col2:
        st.markdown("**Treatment (B)**")
        trt_n_live = st.number_input(
            "Total trials (B)", min_value=1, value=1500, step=50,
            key="live_trt_n",
        )
        trt_s_live = st.number_input(
            "Successes (B)", min_value=0, max_value=int(trt_n_live),
            value=min(115, int(trt_n_live)), step=10, key="live_trt_s",
        )

    try:
        result = msprt_beta_binomial(
            control_successes=ctrl_s_live,
            control_total=ctrl_n_live,
            treatment_successes=trt_s_live,
            treatment_total=trt_n_live,
            alpha_signif=alpha_signif,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )

        # Headline decision
        if result.reject_null:
            st.success(
                f"**STOP — significant difference detected.** "
                f"Likelihood ratio Λ = {result.likelihood_ratio:.2f} ≥ "
                f"threshold {result.rejection_threshold:.1f}."
            )
        else:
            ratio_to_threshold = result.likelihood_ratio / result.rejection_threshold
            if ratio_to_threshold > 0.5:
                pct = ratio_to_threshold * 100
                st.warning(
                    f"**Continue collecting data.** "
                    f"Λ = {result.likelihood_ratio:.2f}, which is "
                    f"{pct:.0f}% of the rejection threshold "
                    f"{result.rejection_threshold:.1f}. Trending toward "
                    f"significance."
                )
            else:
                st.info(
                    f"**Continue collecting data.** "
                    f"Λ = {result.likelihood_ratio:.4f}, well below the "
                    f"rejection threshold {result.rejection_threshold:.1f}. "
                    f"Evidence so far is consistent with no effect."
                )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("p̂_A", f"{result.p_hat_a:.4f}")
        m2.metric(
            "p̂_B",
            f"{result.p_hat_b:.4f}",
            delta=f"{(result.p_hat_b - result.p_hat_a)*100:+.2f}pp",
        )
        m3.metric("Likelihood ratio", f"{result.likelihood_ratio:.3f}")
        m4.metric(
            "log Λ vs threshold",
            f"{result.log_likelihood_ratio:+.2f}",
            delta=f"{result.log_likelihood_ratio - result.log_rejection_threshold:+.2f} vs cutoff",
        )

    except ValueError as e:
        st.error(f"Invalid input: {e}")


# ------------------------------------------------------------
# TAB 2: OPERATING CHARACTERISTICS
# ------------------------------------------------------------
with tab_oc:
    st.subheader("How does mSPRT behave under different scenarios?")
    st.markdown(
        "Simulate many hypothetical experiments under a chosen true effect. "
        "See where mSPRT stops, how often it rejects, and visualize that "
        "peeking does NOT inflate the false-positive rate when there is no "
        "true effect."
    )

    col1, col2 = st.columns(2)
    with col1:
        baseline_oc = st.slider(
            "True control rate p_A",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
            format="%.2f",
        )
        true_lift_pp = st.slider(
            "True lift (percentage points)",
            min_value=-5.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="0.0 means H0 is true (no real effect). Positive means "
                 "treatment is truly better.",
        )
    with col2:
        n_max_oc = st.select_slider(
            "Max sample size per arm",
            options=[500, 1000, 2000, 5000, 10_000],
            value=2000,
        )
        n_sims_oc = st.select_slider(
            "Number of simulated experiments",
            options=[50, 100, 200, 500, 1000],
            value=200,
            help="More sims = smoother curves but slower.",
        )
        check_every_oc = st.select_slider(
            "Peek every N observations",
            options=[1, 5, 10, 25, 50, 100],
            value=25,
        )

    treatment_rate = baseline_oc + true_lift_pp / 100
    if treatment_rate < 0 or treatment_rate > 1:
        st.error(
            f"True treatment rate {treatment_rate:.3f} out of [0, 1]. "
            f"Reduce the lift or change the baseline."
        )
    else:
        st.caption(
            f"Simulating: p_A = {baseline_oc:.3f}, p_B = {treatment_rate:.3f}. "
            f"{n_sims_oc} experiments, up to {n_max_oc} obs/arm, "
            f"peeking every {check_every_oc} obs."
        )

        run_button = st.button("Run simulation", type="primary")

        if run_button:
            progress = st.progress(0)
            status = st.empty()
            stopping_times = []
            rejections = 0
            sample_trajectories = []  # store a few for plotting

            rng = np.random.default_rng(42)

            for i in range(n_sims_oc):
                control_data = rng.binomial(1, baseline_oc, n_max_oc)
                treatment_data = rng.binomial(1, treatment_rate, n_max_oc)

                history = msprt_sequential(
                    control_data,
                    treatment_data,
                    alpha_signif=alpha_signif,
                    prior_alpha=prior_alpha,
                    prior_beta=prior_beta,
                    check_every=check_every_oc,
                )
                final = history[-1]
                stopping_times.append(final.n_a)
                if final.reject_null:
                    rejections += 1

                # Save first 30 trajectories for the LR plot
                if i < 30:
                    sample_trajectories.append(
                        [(h.n_a, h.log_likelihood_ratio) for h in history]
                    )

                if (i + 1) % max(1, n_sims_oc // 20) == 0:
                    progress.progress((i + 1) / n_sims_oc)
                    status.text(f"Simulated {i+1}/{n_sims_oc} experiments...")

            progress.empty()
            status.empty()

            rejection_rate = rejections / n_sims_oc
            mean_stop = np.mean(stopping_times)
            median_stop = np.median(stopping_times)

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Rejection rate",
                f"{rejection_rate:.1%}",
                help="Fraction of simulated experiments that crossed the "
                     "rejection threshold. Under H0 (lift=0) this is "
                     "Type I error; under H1 this is power.",
            )
            m2.metric("Mean stopping time", f"{mean_stop:.0f}")
            m3.metric("Median stopping time", f"{median_stop:.0f}")

            # Interpret
            if abs(true_lift_pp) < 0.01:
                if rejection_rate <= alpha_signif * 1.5:
                    st.success(
                        f"**Type I error is controlled.** "
                        f"True effect = 0, rejection rate = {rejection_rate:.2%} "
                        f"≤ α + slack. mSPRT correctly avoids false positives "
                        f"despite {n_max_oc // check_every_oc} peeks per sim."
                    )
                else:
                    st.warning(
                        f"Rejection rate {rejection_rate:.2%} above α={alpha_signif}. "
                        f"Within Monte Carlo error for small n_sims, but worth "
                        f"investigating if it persists."
                    )
            else:
                st.info(
                    f"With a true lift of {true_lift_pp:+.1f}pp, mSPRT rejected "
                    f"{rejection_rate:.0%} of the time. This is the test's "
                    f"**power** under this alternative."
                )

            # LR trajectory plot
            st.subheader("Likelihood ratio trajectories")
            st.caption(
                "Each line is one simulated experiment. The red line is the "
                "rejection threshold. Trajectories that cross above the "
                "threshold trigger early stopping."
            )

            fig_traj = go.Figure()
            for traj in sample_trajectories:
                xs = [pt[0] for pt in traj]
                ys = [pt[1] for pt in traj]
                fig_traj.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines",
                        line=dict(width=1), opacity=0.4,
                        showlegend=False,
                    )
                )
            fig_traj.add_hline(
                y=-np.log(alpha_signif),
                line=dict(color="red", width=2, dash="dash"),
                annotation_text="Reject threshold",
                annotation_position="top right",
            )
            fig_traj.add_hline(
                y=0,
                line=dict(color="gray", width=1, dash="dot"),
            )
            fig_traj.update_layout(
                xaxis_title="Observations per arm",
                yaxis_title="log Λ",
                height=420,
                showlegend=False,
            )
            st.plotly_chart(fig_traj, use_container_width=True)

            # Stopping time distribution
            st.subheader("Distribution of stopping times")
            fig_stop = go.Figure()
            fig_stop.add_trace(
                go.Histogram(
                    x=stopping_times, nbinsx=30, opacity=0.75,
                )
            )
            fig_stop.add_vline(
                x=mean_stop,
                line=dict(dash="dash"),
                annotation_text=f"Mean: {mean_stop:.0f}",
            )
            fig_stop.update_layout(
                xaxis_title="Observations at stop (per arm)",
                yaxis_title="Number of experiments",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_stop, use_container_width=True)
            st.caption(
                "Experiments that ran to n_max either failed to reach "
                "significance or had no real effect to detect."
            )
        else:
            st.caption("Click 'Run simulation' to generate operating characteristics.")