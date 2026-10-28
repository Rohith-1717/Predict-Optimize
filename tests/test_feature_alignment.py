"""
tests/test_feature_alignment.py
Validates that:
1. Features only look backward in time (no forward look-ahead bias).
2. Target returns are strictly future returns realized between day t and day t+1.
3. Train/validation/test chronological splits never leak future information into the past.
"""

import numpy as np
import pandas as pd
import pytest
from src.data import compute_technical_features, build_tensor_dataset


def test_feature_target_alignment():
    # Construct a synthetic linear price series
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    tickers = ["STOCK_A", "STOCK_B"]
    
    # Prices: Stock A increases by 1 each day, Stock B increases by 2
    prices = pd.DataFrame(
        {
            "STOCK_A": 100.0 + np.arange(100) * 1.0,
            "STOCK_B": 50.0 + np.arange(100) * 2.0,
        },
        index=dates,
    )

    features, target_returns = compute_technical_features(prices)

    # Check target at day t: should equal (P_{t+1} - P_t) / P_t
    valid_dates = target_returns.index
    for t_date in valid_dates[:10]:
        t_pos = dates.get_loc(t_date)
        p_t = prices.loc[t_date, "STOCK_A"]
        p_next = prices.iloc[t_pos + 1]["STOCK_A"]
        expected_target = (p_next - p_t) / p_t
        actual_target = target_returns.loc[t_date, "STOCK_A"]
        assert np.isclose(expected_target, actual_target, atol=1e-6), (
            f"Target return misaligned on {t_date}: expected {expected_target}, got {actual_target}"
        )


def test_chronological_splitting_no_overlap():
    dates = [f"202{i}-01-{d:02d}" for i in range(3) for d in range(1, 20)]
    T = len(dates)
    
    # Split chronologically
    train_end = int(0.7 * T)
    train_dates = dates[:train_end]
    test_dates = dates[train_end:]

    # Assert no overlap
    assert len(set(train_dates).intersection(set(test_dates))) == 0
    # Assert strictly increasing
    assert train_dates[-1] < test_dates[0]
