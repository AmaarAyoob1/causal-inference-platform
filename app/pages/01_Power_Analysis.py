"""
Power Analysis page.

Two workflows:
1. Solve for sample size (given effect, alpha, power -> n)
2. Solve for MDE (given n, alpha, power -> minimum detectable effect)

Supports both proportion-based outcomes (conversion rates) and continuous
outcomes (means with Cohen's d).

Design choices:
- Sensible defaults so users see meaningful output immediately
- Live validation on all inputs
- Sensitivity curve so the user sees how n scales with effect — this is
  the most underrated feature of a sizing tool and the thing that turns
  a "calculator" into a "decision-support tool"
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.power_analysis import (
    sample_size_for_proportions,
    mde_for_proportions,
    sample_size_for_means,
)


st.set_page_config(page_title="Power Analysis", page_icon="🔬", layout="wide")
st.title("🔬 Power Analysis")
st.markdown(
    "Plan your experiment before you run it. Determine the sample size you "
    "need to reliably detect an effect, or the smallest effect you can detect "
    "with the sample size you have."
)

# ============================================================
# SIDEBAR: shared educational context
# ============================================================
with st.sidebar:
    st.header("About power analysis")
    st.markdown(
        """
        **Four quantities, fix any three, the fourth is determined:**
        - Effect size (how big an effect?)
        - Sample size (per group)
        - Alpha (false positive rate)
        - Power (true positive rate)

        Default convention: alpha=0.05, power=0.80.
        These are starting points, not laws — calibrate to your context.
        """
    )
    st.divider()
    st.caption(
        "Common mistake: underpowered experiments produce ambiguous "
        "non-significant results. Common waste: overpowered experiments "
        "detect 'significant' effects too small to matter."
    )


# ============================================================
# OUTCOME TYPE TABS
# ============================================================
tab_prop, tab_cont = st.tabs(["Proportion (conversion rate)", "Continuous (means)"])


# ------------------------------------------------------------
# TAB 1: PROPORTIONS
# ------------------------------------------------------------
with tab_prop:
    st.subheader("Binary outcome (e.g., conversion rate, click-through)")

    mode = st.radio(
        "What do you want to solve for?",
        options=["Sample size (given effect)", "MDE (given sample size)"],
        horizontal=True,
        key="prop_mode",
    )

    col1, col2 = st.columns(2)
    with col1:
        baseline = st.number_input(
            "Baseline conversion rate",
            min_value=0.0001,
            max_value=0.9999,
            value=0.05,
            step=0.005,
            format="%.4f",
            help=(
                "Expected control-arm conversion rate. Lower baselines need "
                "MUCH larger samples — the variance term p(1-p) interacts "
                "with the absolute effect."
            ),
        )

    with col2:
        alpha = st.number_input(
            "Alpha (significance level)",
            min_value=0.001,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="%.3f",
        )
        power = st.number_input(
            "Power (1 - beta)",
            min_value=0.50,
            max_value=0.99,
            value=0.80,
            step=0.05,
            format="%.2f",
        )

    st.divider()

    if mode == "Sample size (given effect)":
        lift_pct = st.slider(
            "Minimum detectable lift (relative %)",
            min_value=0.5,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help=(
                "Relative lift over baseline. 5% means treatment converts "
                "5% better than control (e.g., 5.00% -> 5.25% at baseline=5%)."
            ),
        )
        lift = lift_pct / 100

        try:
            result = sample_size_for_proportions(
                baseline_rate=baseline,
                minimum_detectable_lift=lift,
                alpha=alpha,
                power=power,
            )
            treatment_rate = baseline * (1 + lift)

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Sample size per group",
                f"{result.sample_size_per_group:,}",
            )
            m2.metric(
                "Total sample size",
                f"{result.sample_size_per_group * 2:,}",
            )
            m3.metric(
                "Treatment rate to detect",
                f"{treatment_rate*100:.3f}%",
                delta=f"+{lift*100:.2f}% relative",
            )

            st.success(
                f"To detect a {lift_pct:.1f}% relative lift on a "
                f"{baseline*100:.2f}% baseline at alpha={alpha} and "
                f"power={power*100:.0f}%, you need "
                f"**{result.sample_size_per_group:,} users per group** "
                f"({result.sample_size_per_group * 2:,} total)."
            )

            st.subheader("Sensitivity: sample size vs effect size")
            st.caption(
                "How required sample size changes if your target lift differs "
                "from what you specified. Effect appears squared in the formula, "
                "so halving the lift roughly quadruples the sample needed."
            )

            lift_range = np.linspace(max(0.5, lift_pct * 0.3), lift_pct * 3, 30) / 100
            sample_sizes = []
            for trial_lift in lift_range:
                try:
                    r = sample_size_for_proportions(
                        baseline_rate=baseline,
                        minimum_detectable_lift=trial_lift,
                        alpha=alpha,
                        power=power,
                    )
                    sample_sizes.append(r.sample_size_per_group)
                except ValueError:
                    sample_sizes.append(None)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=lift_range * 100,
                    y=sample_sizes,
                    mode="lines+markers",
                    line=dict(width=2),
                    name="Sample size per group",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[lift_pct],
                    y=[result.sample_size_per_group],
                    mode="markers",
                    marker=dict(size=14, color="red", symbol="x"),
                    name="Your selected lift",
                )
            )
            fig.update_layout(
                xaxis_title="Relative lift (%)",
                yaxis_title="Sample size per group",
                yaxis_type="log",
                height=400,
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Y-axis is logarithmic. Note how steeply the curve climbs as "
                "lift decreases — small effects are expensive to detect."
            )

        except ValueError as e:
            st.error(f"Invalid input: {e}")

    else:  # MDE mode
        n_per_group = st.number_input(
            "Sample size per group (already available)",
            min_value=2,
            max_value=10_000_000,
            value=10_000,
            step=1000,
        )

        try:
            result = mde_for_proportions(
                baseline_rate=baseline,
                sample_size_per_group=n_per_group,
                alpha=alpha,
                power=power,
            )

            phi1 = np.arcsin(np.sqrt(baseline))
            phi2 = phi1 + result.effect_size / 2
            treatment_rate = np.sin(phi2) ** 2
            implied_lift = (treatment_rate - baseline) / baseline

            m1, m2, m3 = st.columns(3)
            m1.metric("Sample size per group", f"{n_per_group:,}")
            m2.metric("Minimum detectable lift", f"{implied_lift*100:.2f}%")
            m3.metric("MDE treatment rate", f"{treatment_rate*100:.3f}%")

            st.info(
                f"With {n_per_group:,} users per group at {baseline*100:.2f}% "
                f"baseline, you can detect a relative lift as small as "
                f"**{implied_lift*100:.2f}%** with {power*100:.0f}% power. "
                f"Effects smaller than this will likely produce non-significant "
                f"results even if real."
            )

        except ValueError as e:
            st.error(f"Invalid input: {e}")


# ------------------------------------------------------------
# TAB 2: CONTINUOUS
# ------------------------------------------------------------
with tab_cont:
    st.subheader("Continuous outcome (e.g., revenue per user, time on page)")
    st.markdown(
        "Specify effect size as **Cohen's d** — the standardized mean "
        "difference. Benchmarks: 0.2 = small, 0.5 = medium, 0.8 = large."
    )

    col1, col2 = st.columns(2)
    with col1:
        cohens_d = st.number_input(
            "Cohen's d",
            min_value=0.01,
            max_value=3.0,
            value=0.50,
            step=0.05,
            format="%.2f",
            help="(mean_treatment - mean_control) / pooled_std_dev",
        )
    with col2:
        alpha_c = st.number_input(
            "Alpha",
            min_value=0.001,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="%.3f",
            key="cont_alpha",
        )
        power_c = st.number_input(
            "Power",
            min_value=0.50,
            max_value=0.99,
            value=0.80,
            step=0.05,
            format="%.2f",
            key="cont_power",
        )

    try:
        result = sample_size_for_means(
            cohens_d=cohens_d, alpha=alpha_c, power=power_c
        )

        m1, m2 = st.columns(2)
        m1.metric(
            "Sample size per group",
            f"{result.sample_size_per_group:,}",
        )
        m2.metric(
            "Total sample size",
            f"{result.sample_size_per_group * 2:,}",
        )

        st.success(
            f"To detect Cohen's d = {cohens_d:.2f} at alpha={alpha_c} and "
            f"power={power_c*100:.0f}%, you need "
            f"**{result.sample_size_per_group:,} per group**."
        )

        st.subheader("Sensitivity: sample size vs effect size")
        d_range = np.linspace(max(0.05, cohens_d * 0.3), cohens_d * 2.5, 40)
        ns = [
            sample_size_for_means(cohens_d=d, alpha=alpha_c, power=power_c).sample_size_per_group
            for d in d_range
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d_range, y=ns, mode="lines", line=dict(width=2)))
        fig.add_trace(
            go.Scatter(
                x=[cohens_d],
                y=[result.sample_size_per_group],
                mode="markers",
                marker=dict(size=14, color="red", symbol="x"),
                name="Selected d",
            )
        )
        fig.update_layout(
            xaxis_title="Cohen's d",
            yaxis_title="Sample size per group",
            yaxis_type="log",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    except ValueError as e:
        st.error(f"Invalid input: {e}")