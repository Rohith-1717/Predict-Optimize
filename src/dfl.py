"""
src/dfl.py
Decision-Focused Learning (DFL) and Predict-Then-Optimize (PTO) training routines.
Implements:
- Standard MSE training (for PTO and U-PTO)
- High-speed Batched Portfolio Decision Loss training (for DFL and U-DFL)
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import MLPEnsemble
from src.portfolio import DifferentiablePortfolioLayer


def train_pto_ensemble(
    model: MLPEnsemble,
    X_train: np.ndarray,      # (T_train, N, F)
    y_train: np.ndarray,      # (T_train, N)
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Dict[str, list]:
    """
    Train MLP Ensemble using standard Mean Squared Error (MSE) loss.
    Used for Strategy 2 (PTO) and Strategy 4 (U-PTO).
    """
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    T_train, N, F = X_train.shape
    history = {"loss": []}

    X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        perm = torch.randperm(T_train)
        epoch_losses = []

        for start_idx in range(0, T_train, batch_size):
            batch_indices = perm[start_idx : start_idx + batch_size]
            X_b = X_tensor[batch_indices]  # (B, N, F)
            y_b = y_tensor[batch_indices]  # (B, N)

            optimizer.zero_grad()
            mean_pred, uncertainty, member_preds = model(X_b)
            # member_preds shape: (M, B, N)
            
            loss = 0.0
            for m in range(model.n_members):
                loss += criterion(member_preds[m], y_b)
            loss = loss / model.n_members

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        history["loss"].append(mean_loss)

    return history


def train_dfl_ensemble(
    model: MLPEnsemble,
    X_train: np.ndarray,      # (T_train, N, F)
    y_train: np.ndarray,      # (T_train, N)
    covs_train: np.ndarray,   # (T_train, N, N)
    risk_aversion: float = 2.5,
    uncertainty_penalty: float = 0.0,
    max_weight: float = 0.15,
    transaction_cost_bps: float = 15.0,
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    aux_mse_weight: float = 0.10,
    device: Optional[torch.device] = None,
) -> Dict[str, list]:
    r"""
    Train MLP Ensemble using Decision-Focused Learning (DFL).
    Backpropagates through the Batched Differentiable Portfolio Layer.
    Objective:
      Loss = - (r_{t+1}^T w_t^*) + 0.5 * lambda * (w_t^*^T Sigma_t w_t^*) + c * ||w_t^* - w_{t-1}^*||_1
             + aux_mse_weight * MSE(pred, true)
    Used for Strategy 3 (DFL, gamma=0) and Strategy 5 (U-DFL, gamma > 0).
    """
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.train()

    portfolio_layer = DifferentiablePortfolioLayer(
        risk_aversion=risk_aversion,
        uncertainty_penalty=uncertainty_penalty,
        max_weight=max_weight,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_criterion = nn.MSELoss()

    T_train, N, F = X_train.shape
    history = {"loss": [], "decision_loss": [], "mse_loss": []}

    X_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    covs_tensor = torch.as_tensor(covs_train, dtype=torch.float32, device=device)
    cost_coeff = transaction_cost_bps / 10000.0

    for epoch in range(epochs):
        # Sample mini-batches of contiguous chunks or random slices
        chunk_starts = list(range(0, T_train - batch_size + 1, batch_size))
        if not chunk_starts:
            chunk_starts = [0]
        np.random.shuffle(chunk_starts)

        epoch_tot_losses = []
        epoch_dec_losses = []
        epoch_mse_losses = []

        for start_idx in chunk_starts:
            end_idx = min(start_idx + batch_size, T_train)
            B_actual = end_idx - start_idx
            if B_actual < 2:
                continue

            X_b = X_tensor[start_idx:end_idx]      # (B, N, F)
            y_b = y_tensor[start_idx:end_idx]      # (B, N)
            cov_b = covs_tensor[start_idx:end_idx]  # (B, N, N)

            optimizer.zero_grad()

            mean_pred, uncertainty, member_preds = model(X_b)  # (B, N)
            u_arg = uncertainty if uncertainty_penalty > 0.0 else None

            # Batched forward solve through differentiable portfolio layer
            w_opt = portfolio_layer(mean_pred, cov_b, u_arg)   # (B, N)

            # 1. Realized portfolio return loss: - sum(w * y)
            ret_loss = -torch.mean(torch.sum(w_opt * y_b, dim=-1))

            # 2. Realized risk penalty: 0.5 * lambda * w^T Sigma w
            quad_risk = torch.bmm(cov_b, w_opt.unsqueeze(-1)).squeeze(-1)  # (B, N)
            risk_loss = 0.5 * risk_aversion * torch.mean(torch.sum(w_opt * quad_risk, dim=-1))

            # 3. Turnover penalty between consecutive time steps
            turnover_loss = cost_coeff * torch.mean(torch.sum(torch.abs(w_opt[1:] - w_opt[:-1]), dim=-1))

            dec_loss = ret_loss + risk_loss + turnover_loss

            # 4. Auxiliary MSE loss
            mse_loss = 0.0
            for m in range(model.n_members):
                mse_loss = mse_loss + mse_criterion(member_preds[m], y_b)
            mse_loss = mse_loss / model.n_members

            total_loss = dec_loss + aux_mse_weight * mse_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_tot_losses.append(total_loss.item())
            epoch_dec_losses.append(dec_loss.item())
            epoch_mse_losses.append(mse_loss.item())

        history["loss"].append(float(np.mean(epoch_tot_losses)) if epoch_tot_losses else 0.0)
        history["decision_loss"].append(float(np.mean(epoch_dec_losses)) if epoch_dec_losses else 0.0)
        history["mse_loss"].append(float(np.mean(epoch_mse_losses)) if epoch_mse_losses else 0.0)

    return history
