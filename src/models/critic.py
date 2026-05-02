"""Critic head."""

from __future__ import annotations

from torch import nn


class CriticHead(nn.Module):
    """Estimate the state value."""

    def __init__(self, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)
