"""
tests/test_portfolio_constraints.py
Validates that portfolio optimization strictly respects:
1. Long-only constraint (w_i >= 0)
2. Budget / full-investment constraint (sum(w_i) == 1)
3. Maximum weight limit per stock (w_i <= max_weight)
"""

import numpy as np
import pytest
from src.portfolio import solve_qp_cvxpy


def test_portfolio_constraints_basic():
    N = 20
    np.random.seed(42)
    expected_returns = np.random.uniform(-0.02, 0.04, size=N)
    
    # Construct random positive definite covariance matrix
    A = np.random.randn(N, N)
    cov = (A @ A.T) * 0.0001 + np.eye(N) * 0.001
    
    max_weight = 0.15
    w = solve_qp_cvxpy(
        expected_returns=expected_returns,
        covariance_matrix=cov,
        uncertainty=None,
        risk_aversion=2.5,
        uncertainty_penalty=0.0,
        max_weight=max_weight,
    )

    # 1. Dimension
    assert len(w) == N
    # 2. Long-only constraint
    assert np.all(w >= -1e-6), f"Negative weights detected: min {w.min()}"
    # 3. Sums to 1
    assert np.isclose(np.sum(w), 1.0, atol=1e-4), f"Weights sum to {np.sum(w)}, expected 1.0"
    # 4. Max weight constraint
    assert np.all(w <= max_weight + 1e-4), f"Weight exceeds max limit: max {w.max()}"


def test_portfolio_constraints_with_uncertainty():
    N = 20
    np.random.seed(123)
    expected_returns = np.random.uniform(-0.01, 0.05, size=N)
    A = np.random.randn(N, N)
    cov = (A @ A.T) * 0.0001 + np.eye(N) * 0.001
    uncertainty = np.random.uniform(0.01, 0.08, size=N)

    max_weight = 0.12
    w = solve_qp_cvxpy(
        expected_returns=expected_returns,
        covariance_matrix=cov,
        uncertainty=uncertainty,
        risk_aversion=3.0,
        uncertainty_penalty=2.0,
        max_weight=max_weight,
    )

    assert np.all(w >= -1e-6)
    assert np.isclose(np.sum(w), 1.0, atol=1e-4)
    assert np.all(w <= max_weight + 1e-4)


def test_uncertainty_penalty_reduces_high_uncertainty_allocation():
    """Assets with artificially spiked uncertainty should receive lower portfolio weight."""
    N = 5
    expected_returns = np.array([0.03, 0.03, 0.03, 0.03, 0.03])
    cov = np.eye(N) * 0.0004
    # Asset 0 has massive uncertainty disagreement
    uncertainty = np.array([0.50, 0.01, 0.01, 0.01, 0.01])

    w_no_u = solve_qp_cvxpy(expected_returns, cov, uncertainty=None, uncertainty_penalty=0.0, max_weight=0.30)
    w_with_u = solve_qp_cvxpy(expected_returns, cov, uncertainty=uncertainty, uncertainty_penalty=5.0, max_weight=0.30)

    assert w_with_u[0] < w_no_u[0], f"Uncertain asset weight did not decrease: {w_with_u[0]} vs {w_no_u[0]}"
