"""
Causal Inference & A/B Testing Platform — Home

Streamlit auto-discovers files in app/pages/ as additional pages.
This file is the landing page.
"""
import streamlit as st

st.set_page_config(
    page_title="Causal Inference Platform",
    page_icon="📊",
    layout="wide",
)

st.title("Causal Inference & A/B Testing Platform")
st.markdown(
    """
    A rigorous toolkit for designing, analyzing, and interpreting
    randomized experiments and observational studies.

    ### What this platform covers

    **Pillar 1 — Pre-experiment design**
    Power analysis, minimum detectable effect estimation, pre-registration.

    **Pillar 2 — Experiment analysis**
    Frequentist tests, Bayesian A/B testing, sequential testing, multiple
    comparison adjustments, CUPED variance reduction.

    **Pillar 3 — Observational causal inference**
    Difference-in-Differences, Propensity Score Matching.

    ---

    Use the navigation sidebar on the left to access individual tools.
    Each tool is built on validated statistical methods with simulation-based
    correctness checks. See the README in the repo for methodology and
    validation details.
    """
)

st.info(
    "**Status:** Power Analysis is live. Frequentist tests, Bayesian A/B, "
    "and observational methods are in development."
)