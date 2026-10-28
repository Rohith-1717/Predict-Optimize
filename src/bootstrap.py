"""
src/bootstrap.py
Block-bootstrap statistical comparison of portfolio strategies.
Preserves temporal autocorrelation and volatility clustering to compute
empirical p-values and confidence intervals for Sharpe ratio differences.
"""

from typing import Dict, Any, List
import numpy as np


def block_bootstrap_sharpe_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    block_length: int = 10,
    n_bootstrap: int = 1000,
    risk_free_rate: float = 0.06,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Circular block bootstrap test for the difference in annualized Sharpe ratios
    between Strategy A (e.g. U-DFL) and Strategy B (e.g. PTO or Equal Weight).

    H0: Sharpe(A) <= Sharpe(B)
    H1: Sharpe(A) > Sharpe(B)
    """
    np.random.seed(seed)
    T = len(returns_a)
    assert len(returns_b) == T, "Return series must have equal length"

    rf_daily = ((1.0 + risk_free_rate) ** (1.0 / 252.0)) - 1.0

    def compute_sharpe(r: np.ndarray) -> float:
        vol = np.std(r, ddof=1)
        if vol < 1e-8:
            return 0.0
        return float((np.mean(r - rf_daily) / vol) * np.sqrt(252.0))

    observed_sharpe_a = compute_sharpe(returns_a)
    observed_sharpe_b = compute_sharpe(returns_b)
    observed_diff = observed_sharpe_a - observed_sharpe_b

    # Generate circular blocks
    num_blocks = int(np.ceil(T / block_length))
    boot_diffs = []

    for _ in range(n_bootstrap):
        start_indices = np.random.randint(0, T, size=num_blocks)
        boot_idx = []
        for s in start_indices:
            boot_idx.extend([(s + k) % T for k in range(block_length)])
        boot_idx = np.array(boot_idx[:T])

        sh_a = compute_sharpe(returns_a[boot_idx])
        sh_b = compute_sharpe(returns_b[boot_idx])
        boot_diffs.append(sh_a - sh_b)

    boot_diffs = np.array(boot_diffs)
    # Empirical one-sided p-value: fraction of bootstrap samples where diff <= 0
    p_value = float(np.mean(boot_diffs <= 0.0))
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return {
        "sharpe_a": float(observed_sharpe_a),
        "sharpe_b": float(observed_sharpe_b),
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_95": [ci_lower, ci_upper],
        "is_significant_5pct": bool(p_value < 0.05),
    }
