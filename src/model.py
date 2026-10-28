"""
src/model.py
Small MLP Ensemble architecture for predicting next-day returns
and estimating predictive uncertainty via ensemble disagreement.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn


class MLPMember(nn.Module):
    """
    A single lightweight Multi-Layer Perceptron member for stock return prediction.
    Designed for fast CPU training and stable gradients.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 32, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (..., input_dim)
        output shape: (..., )
        """
        return self.net(x).squeeze(-1)


class MLPEnsemble(nn.Module):
    """
    Deep Ensemble of small MLPs.
    Predicts next-day returns (ensemble mean) and estimates epistemic
    predictive uncertainty from the disagreement (standard deviation) between members.
    """

    def __init__(
        self,
        n_members: int = 5,
        input_dim: int = 6,
        hidden_dim: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_members = n_members
        self.members = nn.ModuleList([
            MLPMember(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(n_members)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        x shape: (batch_size, n_assets, n_features) or (n_assets, n_features)

        Returns:
        - mean_pred: (batch_size, n_assets) or (n_assets,) - Ensemble mean prediction \hat{r}_t
        - uncertainty: (batch_size, n_assets) or (n_assets,) - Disagreement standard deviation u_t
        - member_preds: (n_members, batch_size, n_assets) or (n_members, n_assets)
        """
        member_preds = torch.stack([member(x) for member in self.members], dim=0)
        # member_preds has shape (n_members, ..., n_assets)

        mean_pred = torch.mean(member_preds, dim=0)
        # Predictive uncertainty = sample standard deviation across ensemble members + eps
        diff_sq = (member_preds - mean_pred.unsqueeze(0)) ** 2
        uncertainty = torch.sqrt(torch.mean(diff_sq, dim=0) + 1e-7)

        return mean_pred, uncertainty, member_preds

    def predict_numpy(self, x_np) -> Tuple:
        """Helper for numpy inputs during backtest."""
        self.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(x_np, dtype=torch.float32)
            mean_pred, uncertainty, _ = self.forward(x_t)
            return mean_pred.cpu().numpy(), uncertainty.cpu().numpy()
