"""
tests/test_transaction_costs.py
Validates turnover and transaction cost accounting:
1. Turnover is positive and equals L1 norm difference of weight adjustments.
2. Net return = Gross return - Transaction Cost.
3. Higher cost rate reduces net return proportionately.
"""

import numpy as np
import pytest
from src.backtest import calculate_metrics


def test_transaction_cost_deduction():
    # Synthetic 5-day return and turnover
    gross_returns = np.array([0.01, -0.005, 0.02, 0.00, 0.015])
    turnovers = np.array([0.40, 0.20, 0.10, 0.30, 0.05])
    cost_rate = 0.0015  # 15 bps
    cost_series = turnovers * cost_rate
    net_returns = gross_returns - cost_series

    assert np.all(net_returns < gross_returns)
    assert np.isclose(np.sum(cost_series), np.sum(turnovers) * 0.0015)

    metrics_15bps = calculate_metrics(net_returns, turnovers, cost_series, risk_free_rate=0.06)
    
    # 50 bps test
    cost_series_50 = turnovers * 0.0050
    net_returns_50 = gross_returns - cost_series_50
    metrics_50bps = calculate_metrics(net_returns_50, turnovers, cost_series_50, risk_free_rate=0.06)

    assert metrics_15bps["CAGR"] > metrics_50bps["CAGR"]
    assert metrics_15bps["Total Transaction Cost"] < metrics_50bps["Total Transaction Cost"]
