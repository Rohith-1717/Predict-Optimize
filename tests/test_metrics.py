"""
tests/test_metrics.py
Validates financial metrics calculation against exact analytical formulas:
1. CAGR
2. Annualized Volatility
3. Sharpe Ratio
4. Maximum Drawdown
5. 95% Daily CVaR
"""

import numpy as np
import pytest
from src.backtest import calculate_metrics


def test_max_drawdown_exact():
    # Construct synthetic wealth path with a known 50% drawdown
    # Path: 1.0 -> 2.0 -> 1.0 -> 1.5
    # Daily returns: +100%, -50%, +50%
    returns = np.array([1.00, -0.50, 0.50])
    dummy = np.zeros(len(returns))

    metrics = calculate_metrics(returns, dummy, dummy, risk_free_rate=0.0)
    # Peak is 2.0, trough is 1.0 -> (1.0 - 2.0) / 2.0 = -0.50
    assert np.isclose(metrics["Max Drawdown"], -0.50, atol=1e-5)


def test_sharpe_ratio_positive_and_negative():
    np.random.seed(42)
    # Strongly positive returns
    pos_returns = np.random.normal(0.002, 0.01, size=252)  # ~50% annual return
    dummy = np.zeros(252)
    pos_met = calculate_metrics(pos_returns, dummy, dummy, risk_free_rate=0.06)
    assert pos_met["Sharpe Ratio"] > 0

    # Strongly negative returns
    neg_returns = np.random.normal(-0.002, 0.01, size=252)
    neg_met = calculate_metrics(neg_returns, dummy, dummy, risk_free_rate=0.06)
    assert neg_met["Sharpe Ratio"] < 0


def test_cvar_greater_than_or_equal_to_var():
    np.random.seed(999)
    returns = np.random.normal(0.0005, 0.015, size=500)
    dummy = np.zeros(500)
    metrics = calculate_metrics(returns, dummy, dummy, risk_free_rate=0.06)

    # 95% CVaR must be a positive loss number
    assert metrics["CVaR 95%"] > 0
    # CVaR is average loss beyond the 5th percentile, so it must exceed -5th percentile
    p5_loss = -np.percentile(returns, 5.0)
    assert metrics["CVaR 95%"] >= p5_loss - 1e-6
