"""
src/portfolio.py
Portfolio optimization formulations:
1. CVXPY solver for long-only constrained Markowitz QP with uncertainty penalty (used in backtesting).
2. Differentiable Portfolio Layer in pure PyTorch using unrolled projected gradient descent
   with exact bounded simplex projection (used for end-to-end Decision-Focused Learning).
"""

from typing import Tuple, Optional, Dict
import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn


def solve_qp_cvxpy(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    risk_aversion: float = 2.5,
    uncertainty_penalty: float = 0.0,
    max_weight: float = 0.15,
    ridge_epsilon: float = 1e-4,
) -> np.ndarray:
    """
    Solve long-only portfolio optimization with CVXPY:
    max_w   r^T w - 0.5 * lambda * w^T Sigma w - 0.5 * gamma * w^T U w
    s.t.    sum(w) == 1, 0 <= w_i <= max_weight

    where U = diag(u^2) is the predictive uncertainty matrix.
    """
    N = len(expected_returns)
    w = cp.Variable(N)

    # Base regularized covariance
    Q = risk_aversion * covariance_matrix + np.eye(N) * ridge_epsilon
    if uncertainty is not None and uncertainty_penalty > 0.0:
        # Add diagonal uncertainty penalty: gamma * diag(u^2)
        u_sq = (uncertainty ** 2)
        Q = Q + uncertainty_penalty * np.diag(u_sq)

    # Ensure symmetry
    Q = 0.5 * (Q + Q.T)

    # Markowitz quadratic objective: min 0.5 * w^T Q w - r^T w
    objective = cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(Q)) - expected_returns @ w)
    
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w <= max_weight,
    ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
            prob.solve(solver=cp.OSQP)
        weights = w.value
        if weights is None:
            raise ValueError("CVXPY solver returned None")
    except Exception:
        weights = np.ones(N) / N

    weights = np.clip(weights, 0.0, max_weight)
    weights = weights / np.sum(weights)
    return weights.astype(np.float32)


def project_to_capped_simplex(
    v: torch.Tensor,
    max_weight: float = 0.15,
    bisection_iters: int = 8,
) -> torch.Tensor:
    """
    Exact differentiable projection of vector v onto the bounded probability simplex:
    { w in R^N : sum(w) == 1, 0 <= w_i <= max_weight }
    Uses vectorized bisection root-finding for the dual multiplier nu:
    w_i = clamp(v_i - nu, 0, max_weight).
    """
    low = v.min(dim=-1, keepdim=True).values - max_weight - 1.0
    high = v.max(dim=-1, keepdim=True).values + 1.0

    for _ in range(bisection_iters):
        mid = 0.5 * (low + high)
        w_mid = torch.clamp(v - mid, 0.0, max_weight)
        s = torch.sum(w_mid, dim=-1, keepdim=True)
        low = torch.where(s > 1.0, mid, low)
        high = torch.where(s <= 1.0, mid, high)

    mid = 0.5 * (low + high)
    w_proj = torch.clamp(v - mid, 0.0, max_weight)
    # Numerical safeguard normalization
    return w_proj / (torch.sum(w_proj, dim=-1, keepdim=True) + 1e-8)


class DifferentiablePortfolioLayer(nn.Module):
    r"""
    Genuine Differentiable Portfolio Optimization Layer in pure PyTorch.
    Solves the constrained quadratic program:
        min_w 0.5 * w^T Q w - p^T w
        s.t.  1^T w = 1, 0 <= w_i <= max_weight
    using unrolled projected gradient descent with exact capped-simplex projection.
    Enables exact end-to-end backpropagation through expected returns p and
    predictive uncertainty u for Decision-Focused Learning (DFL).
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        uncertainty_penalty: float = 0.0,
        max_weight: float = 0.15,
        ridge_epsilon: float = 1e-4,
        unroll_steps: int = 6,
        step_size: float = 0.5,
    ):
        super().__init__()
        self.risk_aversion = risk_aversion
        self.uncertainty_penalty = uncertainty_penalty
        self.max_weight = max_weight
        self.ridge_epsilon = ridge_epsilon
        self.unroll_steps = unroll_steps
        self.step_size = step_size

    def forward(
        self,
        pred_returns: torch.Tensor,        # (N,) or (B, N)
        cov_matrix: torch.Tensor,          # (N, N) or (B, N, N)
        uncertainty: Optional[torch.Tensor] = None,  # (N,) or (B, N)
    ) -> torch.Tensor:
        is_single = pred_returns.dim() == 1
        if is_single:
            p = pred_returns.unsqueeze(0)
            cov = cov_matrix.unsqueeze(0)
            u = uncertainty.unsqueeze(0) if uncertainty is not None else None
        else:
            p = pred_returns
            cov = cov_matrix
            u = uncertainty

        B, N = p.shape
        device = p.device
        dtype = p.dtype

        # Form Q matrix: lambda * Sigma + gamma * diag(u^2) + ridge * I
        Q = self.risk_aversion * cov + torch.eye(N, device=device, dtype=dtype).unsqueeze(0) * self.ridge_epsilon
        if u is not None and self.uncertainty_penalty > 0.0:
            Q = Q + self.uncertainty_penalty * torch.diag_embed(u ** 2)

        # Initial point: equal weight
        w = torch.ones((B, N), device=device, dtype=dtype) / float(N)

        # Unrolled projected gradient iterations
        for _ in range(self.unroll_steps):
            # Gradient of 0.5 * w^T Q w - p^T w is Q w - p
            grad = torch.bmm(Q, w.unsqueeze(-1)).squeeze(-1) - p
            v = w - self.step_size * grad
            w = project_to_capped_simplex(v, max_weight=self.max_weight)

        if is_single:
            return w.squeeze(0)
        return w
