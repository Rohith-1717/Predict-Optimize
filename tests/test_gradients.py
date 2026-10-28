"""
tests/test_gradients.py
Validates the mathematical correctness and stability of the
Differentiable Portfolio Layer gradients used in Decision-Focused Learning (DFL).
"""

import numpy as np
import pytest
import torch
from src.portfolio import DifferentiablePortfolioLayer


def test_portfolio_layer_gradients_exist_and_finite():
    layer = DifferentiablePortfolioLayer(
        risk_aversion=2.0,
        uncertainty_penalty=1.5,
        max_weight=0.20,
    )

    N = 6
    torch.manual_seed(42)
    p = torch.randn(N, dtype=torch.float32, requires_grad=True)
    u = torch.rand(N, dtype=torch.float32, requires_grad=True)
    cov = torch.eye(N, dtype=torch.float32)

    w = layer(p, cov, u)
    target_returns = torch.tensor([0.02, 0.05, -0.01, 0.03, -0.02, 0.04], dtype=torch.float32)
    
    # Realized downstream task loss
    loss = -torch.sum(w * target_returns)
    loss.backward()

    # Assert gradients exist and are finite
    assert p.grad is not None, "Gradient with respect to predicted returns is None"
    assert u.grad is not None, "Gradient with respect to uncertainty is None"
    assert not torch.isnan(p.grad).any(), "NaN found in predicted return gradients"
    assert not torch.isnan(u.grad).any(), "NaN found in uncertainty gradients"
    assert not torch.isinf(p.grad).any(), "Inf found in predicted return gradients"
    assert not torch.isinf(u.grad).any(), "Inf found in uncertainty gradients"


def test_gradient_direction_positive_expected_return():
    """
    Increasing the expected return p_i of an interior asset should increase its allocation w_i.
    Therefore, maximizing w_i (objective = -w_i) should yield a negative gradient d(-w_i)/dp_i < 0.
    """
    layer = DifferentiablePortfolioLayer(risk_aversion=3.0, uncertainty_penalty=0.0, max_weight=0.50)
    p = torch.tensor([0.02, 0.02, 0.02], dtype=torch.float32, requires_grad=True)
    cov = torch.eye(3, dtype=torch.float32)

    w = layer(p, cov, None)
    # Objective: promote asset 0
    loss = -w[0]
    loss.backward()

    # The gradient d(loss)/dp[0] should be strictly negative (increasing p[0] reduces loss)
    assert p.grad[0].item() < 0.0, f"Expected negative gradient for asset 0, got {p.grad[0].item()}"


def test_gradient_direction_uncertainty_penalty():
    """
    Increasing uncertainty u_i should penalize allocation w_i.
    """
    layer = DifferentiablePortfolioLayer(risk_aversion=2.0, uncertainty_penalty=3.0, max_weight=0.50)
    p = torch.tensor([0.05, 0.02, 0.02], dtype=torch.float32)
    u = torch.tensor([0.10, 0.10, 0.10], dtype=torch.float32, requires_grad=True)
    cov = torch.eye(3, dtype=torch.float32)

    w = layer(p, cov, u)
    loss = -w[0]
    loss.backward()

    # Since asset 0 has high weight, increasing u[0] will penalize it, increasing loss
    # Therefore d(loss)/du[0] should be positive
    assert u.grad[0].item() > 0.0, f"Expected positive gradient for uncertainty on asset 0, got {u.grad[0].item()}"
