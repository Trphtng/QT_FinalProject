"""Sequence encoders for portfolio states."""

from __future__ import annotations

import torch
from torch import nn


class BaseEncoder(nn.Module):
    """Base class for market state encoders."""

    def __init__(self, input_dim: int, hidden_dim: int, n_assets: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_assets = n_assets


class LSTMEncoder(BaseEncoder):
    def __init__(self, input_dim: int, hidden_dim: int, n_assets: int, num_layers: int, dropout: float) -> None:
        super().__init__(input_dim, hidden_dim, n_assets)
        self.lstm = nn.LSTM(
            input_size=input_dim * n_assets,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, market: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_assets, n_features = market.shape
        x = market.reshape(batch, seq_len, n_assets * n_features)
        output, _ = self.lstm(x)
        return output[:, -1, :]


class CNN1DEncoder(BaseEncoder):
    def __init__(self, input_dim: int, hidden_dim: int, n_assets: int, cnn_channels: int, dropout: float) -> None:
        super().__init__(input_dim, hidden_dim, n_assets)
        channels = n_assets * input_dim
        self.net = nn.Sequential(
            nn.Conv1d(channels, cnn_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels, hidden_dim),
        )

    def forward(self, market: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_assets, n_features = market.shape
        x = market.reshape(batch, seq_len, n_assets * n_features).transpose(1, 2)
        return self.net(x)


class TransformerEncoderModel(BaseEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_assets: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(input_dim, hidden_dim, n_assets)
        self.input_proj = nn.Linear(input_dim * n_assets, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, market: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_assets, n_features = market.shape
        x = market.reshape(batch, seq_len, n_assets * n_features)
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.norm(x[:, -1, :])


def build_encoder(
    encoder_type: str,
    input_dim: int,
    hidden_dim: int,
    n_assets: int,
    model_cfg: dict,
) -> nn.Module:
    """Factory for encoder selection."""
    encoder_type = encoder_type.lower()
    if encoder_type == "lstm":
        return LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_assets=n_assets,
            num_layers=int(model_cfg.get("lstm_layers", model_cfg.get("num_layers", 2))),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if encoder_type == "cnn":
        return CNN1DEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_assets=n_assets,
            cnn_channels=int(model_cfg.get("cnn_channels", 64)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if encoder_type == "transformer":
        return TransformerEncoderModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_assets=n_assets,
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_heads=int(model_cfg.get("transformer_heads", 4)),
            ff_dim=int(model_cfg.get("transformer_ff_dim", hidden_dim * 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")
