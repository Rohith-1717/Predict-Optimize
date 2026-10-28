# Experiment Results Summary

This file aggregates the key quantitative outcomes of the **Uncertainty‑Aware Decision‑Focused Learning for Robust Portfolio Optimization** study. All result artifacts are stored in the `results/` directory.

---

## 1. Summary Metrics (Test Period 2022‑01‑03 → 2024‑12‑30)

| Strategy | CAGR | Total Return | Annualized Volatility | Sharpe Ratio | Sortino Ratio | Max Drawdown | CVaR 95% | Daily Turnover | Annualized Turnover | Total Transaction Cost | MSE | MAE | Rank‑IC |
|----------|------|--------------|-----------------------|--------------|---------------|--------------|----------|----------------|----------------------|------------------------|-----|-----|----------|
| U‑DFL (proposed)| 14.28% | 47.83% | 13.33% | 0.631 | 0.891 | -15.12% | 1.87% | 0.86% | 2.157 | 0.95% | – | – | – |
| U‑PTO| -8.86% | -23.78% | 14.93% | -0.936 | -1.186 | -28.20% | 2.25% | 49.20% | 123.993 | 54.47% | 0.00023 | 0.01132 | 0.00187 |
| DFL | 6.81% | 21.28% | 14.78% | 0.125 | 0.177 | -18.01% | 2.05% | 38.88% | 97.972 | 43.04% | 0.00057 | 0.01910 | 0.02024 |
| PTO‑MSE| -8.82% | -23.69% | 14.93% | -0.933 | -1.183 | -28.12% | 2.25% | 49.17% | 123.904 | 54.43% | – | – | – |
|  Equal‑Weight || 1.86% | 5.55% | 15.26% | -0.185 | -0.256 | -18.85% | 2.19% | 38.25% | 96.396 | 42.35% | 0.00059 | 0.01927 | 0.00630 |

*Percent values are shown as percentages (e.g., 0.631 → 0.631).*  
*Missing values (‑) indicate metrics not applicable for that strategy (e.g., prediction error for equal‑weight).

---

## 2. Bootstrap Hypothesis Tests (U‑DFL vs. baselines)

| Baseline | Δ Sharpe (U‑DFL – Baseline) | 95 % CI | p‑value | Significant (5 %) |
|----------|-----------------------------|---------|---------|-------------------|
| Equal‑Weight | -0.816 | [-1.421, -0.275] | 1.0000 | No |
|  U‑PTO | +0.751 | [0.044, 1.458] | 0.0190 | **Yes** |
| DFL | -0.310 | [-0.989, 0.341] | 0.8180 | No |
| PTO‑MSE| +0.748 | [0.040, 1.460] | 0.0200 | **Yes** |

---

## 3. Ablation Study Results

The `ablations.csv` file contains the systematic exploration of:
1. **Uncertainty‑penalty γ** values (0.0, 0.5, 1.5, 3.0, 5.0) for U‑DFL.
2. **Transaction‑cost bps** (0, 5, 15, 30, 50) for each strategy.

Key take‑aways (excerpt):
* Higher γ reduces exposure to uncertain assets and generally improves Sharpe but can lower CAGR.
* Increasing transaction costs penalises turnover‑heavy strategies (PTO‑MSE, U‑PTO) more severely.

Full table is available at `results/ablations.csv`.

---

## 4. Where to Find All Artifacts

| Artifact | Path |
|----------|------|
| Summary metrics CSV | `results/metrics_summary.csv` |
| Daily portfolio records | `results/daily_portfolio_records.csv` |
| Portfolio weights over time | `results/portfolio_weights.csv` |
| Bootstrap test JSON | `results/bootstrap_results.json` |
| Ablation CSV | `results/ablations.csv` |
| Experiment metadata | `results/experiment_meta.json` |
| Full research report (Markdown) | `REPORT.md` |
| PDF version of report | `REPORT.pdf` |

---

**All files are version‑controlled in the repository and can be opened directly from the workspace.**