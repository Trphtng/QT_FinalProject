"""Inference helpers for trained models."""

from __future__ import annotations

from typing import Any

import torch

from src.agents.trainer import PPOTrainer
from src.env.portfolio_env import PortfolioEnv
from src.models.actor_critic import ActorCriticNetwork


def load_trained_model(
    checkpoint_path: str,
    model: ActorCriticNetwork,
    device: torch.device,
) -> ActorCriticNetwork:
    """Load model weights from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: ActorCriticNetwork,
    env: PortfolioEnv,
    trainer_cfg: dict,
    device: torch.device,
) -> dict[str, Any]:
    """Run deterministic inference over a target environment."""
    trainer = PPOTrainer(model=model, train_env=env, val_env=env, cfg=trainer_cfg, device=device)
    return trainer.run_policy(env, deterministic=True)
