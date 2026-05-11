# Causal Inference & A/B Testing Platform

A production-grade statistical platform for rigorous experimentation and causal inference. Built to bridge the gap between toy A/B testing calculators and enterprise-grade tools that cost thousands of dollars per year.

## Why This Exists

Most online A/B test calculators handle only the simplest scenarios. They fail to address:
- **Sequential testing**: Peeking at results inflates false positive rates 5-10x
- **Multiple comparisons**: Running 10 tests guarantees a "significant" result by chance
- **Observational data**: When randomization isn't possible, naive comparisons mislead
- **Variance reduction**: CUPED and stratification can reduce required sample sizes by 50%

This platform implements these methods correctly, making rigorous experimentation accessible to anyone running tests on their products, behaviors, or research.

## Features

### Pre-Experiment Design
- Power analysis with sample size calculation
- Minimum Detectable Effect (MDE) estimation
- Pre-registration form with parameter locking

### Experiment Analysis
- Frequentist tests (t, chi-squared, Mann-Whitney)
- Bayesian A/B testing with credible intervals
- Sequential testing (mSPRT) for peeking-safe analysis
- Multiple comparison adjustments (Bonferroni, BH-FDR, Holm)
- CUPED variance reduction

### Observational Causal Inference
- Difference-in-Differences
- Propensity Score Matching
- Synthetic Control Method
- Instrumental Variables

## Tech Stack

- **Frontend**: Streamlit
- **Statistics**: scipy, statsmodels, pymc
- **Causal Inference**: dowhy, econml, causalimpact
- **Deployment**: Streamlit Cloud

## Status

Active development. Currently building core statistical modules.

## Author

Ayoob Amaar | M.S. Applied Statistics & Machine Learning + M.S. Financial Engineering | Claremont Graduate University

[LinkedIn](https://linkedin.com/in/ayoob-amaar) | [GitHub](https://github.com/AmaarAyoob1)
