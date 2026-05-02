"""Shared actor-critic network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.actor import ActorHead
from src.models.critic import CriticHead
from src.models.encoders import build_encoder


@dataclass
class PolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    concentration: torch.Tensor


class ActorCriticNetwork(nn.Module):
    """Portfolio actor-critic with selectable market encoder."""

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        portfolio_dim: int,
        model_cfg: dict,
    ) -> None:
        super().__init__()
        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        encoder_type = str(model_cfg.get("encoder_type", "transformer"))
        self.encoder = build_encoder(
            encoder_type=encoder_type,
            input_dim=n_features,
            hidden_dim=hidden_dim,
            n_assets=n_assets,
            model_cfg=model_cfg,
        )
        self.portfolio_proj = nn.Sequential(
            nn.Linear(portfolio_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.actor = ActorHead(hidden_dim, n_assets + 1, dropout=float(model_cfg.get("dropout", 0.1)))
        self.critic = CriticHead(hidden_dim, dropout=float(model_cfg.get("dropout", 0.1)))

    def encode(self, market: torch.Tensor, portfolio: torch.Tensor) -> torch.Tensor:
        market_features = self.encoder(market)
        portfolio_features = self.portfolio_proj(portfolio)
        return self.shared(torch.cat([market_features, portfolio_features], dim=-1))

    def forward(self, market: torch.Tensor, portfolio: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
        latent = self.encode(market, portfolio)
        dist = self.actor.distribution(latent)
        if deterministic:
            concentration = dist.concentration
            action = concentration / concentration.sum(dim=-1, keepdim=True)
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(latent)
        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            concentration=dist.concentration,
        )

    def evaluate_actions(self, market: torch.Tensor, portfolio: torch.Tensor, actions: torch.Tensor):
        latent = self.encode(market, portfolio)
        dist = self.actor.distribution(latent)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(latent)
        return log_prob, entropy, value
