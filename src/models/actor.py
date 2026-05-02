"""Actor head for simplex portfolio weights."""

from __future__ import annotations

import torch
from torch import distributions, nn


class ActorHead(nn.Module):
    """Parameterize a Dirichlet distribution over portfolio weights."""

    def __init__(self, input_dim: int, action_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, action_dim),
        )
        self.softplus = nn.Softplus()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        concentration = self.softplus(self.net(features)) + 1e-3
        return concentration

    def distribution(self, features: torch.Tensor) -> distributions.Dirichlet:
        return distributions.Dirichlet(self.forward(features))
