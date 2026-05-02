"""PPO-style trainer for the actor critic portfolio agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from src.env.portfolio_env import PortfolioEnv
from src.models.actor_critic import ActorCriticNetwork
from src.utils.logger import get_logger
from src.utils.metrics import compute_performance_metrics


LOGGER = get_logger(__name__)


@dataclass
class RolloutBatch:
    market: torch.Tensor
    portfolio: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RunningMeanStd:
    """Numerically stable running moments for reward normalization."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        variance = self.m2 / (self.count - 1)
        return float(np.sqrt(max(variance, 1e-8)))

    def normalize(self, value: float) -> float:
        normalized = (value - self.mean) / (self.std + 1e-8)
        return float(np.clip(normalized, -5.0, 5.0))


class PPOTrainer:
    """Train an actor-critic model with clipped PPO objective."""

    def __init__(
        self,
        model: ActorCriticNetwork,
        train_env: PortfolioEnv,
        val_env: PortfolioEnv,
        cfg: dict,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_env = train_env
        self.val_env = val_env
        self.cfg = cfg
        self.device = device
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg.get("weight_decay", 1e-5)),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=int(cfg.get("lr_scheduler_patience", 5)),
            factor=float(cfg.get("lr_scheduler_factor", 0.5)),
        )
        self.amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        self.scaler = torch.amp.GradScaler(
            self.amp_device_type,
            enabled=bool(cfg.get("use_mixed_precision", True) and device.type == "cuda"),
        )
        self.writer = SummaryWriter(log_dir=cfg["tensorboard_dir"])
        self.checkpoint_path = Path(cfg["checkpoint_path"])
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_score = -np.inf
        self.start_epoch = 0
        self.reward_stats = RunningMeanStd()
        self.history: dict[str, list[float]] = {
            "train_reward": [],
            "train_reward_raw": [],
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "entropy_coef": [],
            "val_sharpe": [],
            "val_max_drawdown": [],
            "val_score": [],
            "val_final_value": [],
            "mean_gross_return": [],
            "mean_net_return": [],
            "mean_turnover": [],
            "mean_drawdown": [],
            "mean_variance": [],
            "mean_cost": [],
            "mean_reward_log_term": [],
            "mean_reward_return_bonus": [],
            "mean_penalty_turnover": [],
            "mean_penalty_drawdown": [],
            "mean_penalty_variance": [],
            "mean_reward_sharpe_bonus": [],
            "mean_reward_momentum_bonus": [],
        }

        resume_from = cfg.get("resume_from")
        if resume_from:
            self.load_checkpoint(resume_from)

    def train(self) -> dict[str, list[float]]:
        patience = 0
        total_epochs = int(self.cfg["total_epochs"])
        for epoch in range(self.start_epoch, total_epochs):
            entropy_coef = self._cosine_decay(
                step=epoch,
                total_steps=total_epochs,
                start=float(self.cfg.get("entropy_coef_start", self.cfg.get("entropy_coef", 0.01))),
                end=float(self.cfg.get("entropy_coef_end", 0.001)),
            )
            rollout, raw_reward_sum, rollout_diag = self.collect_rollout(self.train_env)
            train_stats = self.update_policy(rollout, entropy_coef=entropy_coef)
            val_stats = self.evaluate_env(self.val_env)
            val_sharpe = float(val_stats["Sharpe Ratio"])
            val_max_drawdown = abs(float(val_stats["Max Drawdown"]))
            score = val_sharpe - 0.5 * val_max_drawdown
            self.scheduler.step(score)

            self.history["train_reward"].append(float(rollout.rewards.sum().item()))
            self.history["train_reward_raw"].append(raw_reward_sum)
            self.history["actor_loss"].append(train_stats["actor_loss"])
            self.history["critic_loss"].append(train_stats["critic_loss"])
            self.history["entropy"].append(train_stats["entropy"])
            self.history["entropy_coef"].append(entropy_coef)
            self.history["val_sharpe"].append(val_sharpe)
            self.history["val_max_drawdown"].append(val_max_drawdown)
            self.history["val_score"].append(score)
            self.history["val_final_value"].append(float(val_stats["Final Portfolio Value"]))
            self.history["mean_gross_return"].append(rollout_diag["mean_gross_return"])
            self.history["mean_net_return"].append(rollout_diag["mean_net_return"])
            self.history["mean_turnover"].append(rollout_diag["mean_turnover"])
            self.history["mean_drawdown"].append(rollout_diag["mean_drawdown"])
            self.history["mean_variance"].append(rollout_diag["mean_variance"])
            self.history["mean_cost"].append(rollout_diag["mean_cost"])
            self.history["mean_reward_log_term"].append(rollout_diag["mean_reward_log_term"])
            self.history["mean_reward_return_bonus"].append(rollout_diag["mean_reward_return_bonus"])
            self.history["mean_penalty_turnover"].append(rollout_diag["mean_penalty_turnover"])
            self.history["mean_penalty_drawdown"].append(rollout_diag["mean_penalty_drawdown"])
            self.history["mean_penalty_variance"].append(rollout_diag["mean_penalty_variance"])
            self.history["mean_reward_sharpe_bonus"].append(rollout_diag["mean_reward_sharpe_bonus"])
            self.history["mean_reward_momentum_bonus"].append(rollout_diag["mean_reward_momentum_bonus"])

            current_lr = self.optimizer.param_groups[0]["lr"]
            LOGGER.info(
                "Step %s | Reward %.4f | Raw Reward %.4f | Score %.4f | Actor Loss %.4f | Critic Loss %.4f | Entropy %.4f | EntCoef %.5f | LR %.6f",
                epoch + 1,
                self.history["train_reward"][-1],
                self.history["train_reward_raw"][-1],
                score,
                train_stats["actor_loss"],
                train_stats["critic_loss"],
                train_stats["entropy"],
                entropy_coef,
                current_lr,
            )
            LOGGER.info(
                "Gross %.6f | Net %.6f | Cost %.6f",
                rollout_diag["mean_gross_return"],
                rollout_diag["mean_net_return"],
                rollout_diag["mean_cost"],
            )
            LOGGER.info(
                "Turnover %.6f | DD %.6f | Var %.6f",
                rollout_diag["mean_turnover"],
                rollout_diag["mean_drawdown"],
                rollout_diag["mean_variance"],
            )
            LOGGER.info(
                "RLog %.6f | RBonus %.6f | RSharpe %.6f | RMomo %.6f | PTurn %.6f | PDD %.6f | PVar %.6f",
                rollout_diag["mean_reward_log_term"],
                rollout_diag["mean_reward_return_bonus"],
                rollout_diag["mean_reward_sharpe_bonus"],
                rollout_diag["mean_reward_momentum_bonus"],
                rollout_diag["mean_penalty_turnover"],
                rollout_diag["mean_penalty_drawdown"],
                rollout_diag["mean_penalty_variance"],
            )

            self.writer.add_scalar("train/reward", self.history["train_reward"][-1], epoch)
            self.writer.add_scalar("train/reward_raw", self.history["train_reward_raw"][-1], epoch)
            self.writer.add_scalar("train/actor_loss", train_stats["actor_loss"], epoch)
            self.writer.add_scalar("train/critic_loss", train_stats["critic_loss"], epoch)
            self.writer.add_scalar("train/entropy", train_stats["entropy"], epoch)
            self.writer.add_scalar("train/entropy_coef", entropy_coef, epoch)
            self.writer.add_scalar("val/sharpe", val_sharpe, epoch)
            self.writer.add_scalar("val/max_drawdown", val_max_drawdown, epoch)
            self.writer.add_scalar("val/score", score, epoch)
            self.writer.add_scalar("val/final_value", val_stats["Final Portfolio Value"], epoch)
            self.writer.add_scalar("train/lr", current_lr, epoch)
            self.writer.add_scalar("train/mean_gross_return", rollout_diag["mean_gross_return"], epoch)
            self.writer.add_scalar("train/mean_net_return", rollout_diag["mean_net_return"], epoch)
            self.writer.add_scalar("train/mean_turnover", rollout_diag["mean_turnover"], epoch)
            self.writer.add_scalar("train/mean_drawdown", rollout_diag["mean_drawdown"], epoch)
            self.writer.add_scalar("train/mean_variance", rollout_diag["mean_variance"], epoch)
            self.writer.add_scalar("train/mean_cost", rollout_diag["mean_cost"], epoch)
            self.writer.add_scalar("train/mean_reward_log_term", rollout_diag["mean_reward_log_term"], epoch)
            self.writer.add_scalar("train/mean_reward_return_bonus", rollout_diag["mean_reward_return_bonus"], epoch)
            self.writer.add_scalar("train/mean_reward_sharpe_bonus", rollout_diag["mean_reward_sharpe_bonus"], epoch)
            self.writer.add_scalar("train/mean_reward_momentum_bonus", rollout_diag["mean_reward_momentum_bonus"], epoch)
            self.writer.add_scalar("train/mean_penalty_turnover", rollout_diag["mean_penalty_turnover"], epoch)
            self.writer.add_scalar("train/mean_penalty_drawdown", rollout_diag["mean_penalty_drawdown"], epoch)
            self.writer.add_scalar("train/mean_penalty_variance", rollout_diag["mean_penalty_variance"], epoch)

            if score > self.best_score:
                self.best_score = score
                patience = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                patience += 1

            self.save_checkpoint(epoch, is_best=False)
            if patience >= int(self.cfg.get("early_stopping_patience", 25)):
                LOGGER.info("Early stopping triggered at epoch %s", epoch + 1)
                break

        self.writer.flush()
        history_path = self.checkpoint_path.parent / "training_history.json"
        history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        return self.history

    def collect_rollout(self, env: PortfolioEnv) -> tuple[RolloutBatch, float, dict[str, float]]:
        market_list = []
        portfolio_list = []
        action_list = []
        log_prob_list = []
        reward_list = []
        raw_reward_sum = 0.0
        done_list = []
        value_list = []
        gross_returns: list[float] = []
        net_returns: list[float] = []
        turnovers: list[float] = []
        drawdowns: list[float] = []
        variances: list[float] = []
        costs: list[float] = []
        reward_log_terms: list[float] = []
        reward_return_bonus_terms: list[float] = []
        penalty_turnovers: list[float] = []
        penalty_drawdowns: list[float] = []
        penalty_variances: list[float] = []
        reward_sharpe_bonus_terms: list[float] = []
        reward_momentum_bonus_terms: list[float] = []

        state = env.reset()
        done = False
        while not done:
            market = self._tensor(state["market"]).unsqueeze(0)
            portfolio = self._tensor(state["portfolio"]).unsqueeze(0)
            with torch.no_grad():
                output = self.model(market, portfolio, deterministic=False)
            action = output.action.squeeze(0).cpu().numpy()
            next_state, reward, done, info = env.step(action)
            raw_reward_sum += float(reward)
            self.reward_stats.update(float(reward))
            reward = self.reward_stats.normalize(float(reward))
            gross_returns.append(float(info["gross_return"]))
            net_returns.append(float(info["net_return"]))
            turnovers.append(float(info["turnover"]))
            drawdowns.append(float(info["drawdown"]))
            variances.append(float(info["variance"]))
            costs.append(float(info.get("cost", info["transaction_cost"])))
            reward_log_terms.append(float(info.get("reward_log_term", 0.0)))
            reward_return_bonus_terms.append(float(info.get("reward_return_bonus", 0.0)))
            penalty_turnovers.append(float(info.get("penalty_turnover", 0.0)))
            penalty_drawdowns.append(float(info.get("penalty_drawdown", 0.0)))
            penalty_variances.append(float(info.get("penalty_variance", 0.0)))
            reward_sharpe_bonus_terms.append(float(info.get("reward_sharpe_bonus", 0.0)))
            reward_momentum_bonus_terms.append(float(info.get("reward_momentum_bonus", 0.0)))

            market_list.append(market.squeeze(0))
            portfolio_list.append(portfolio.squeeze(0))
            action_list.append(output.action.squeeze(0))
            log_prob_list.append(output.log_prob.squeeze(0))
            reward_list.append(torch.tensor(reward, dtype=torch.float32))
            done_list.append(torch.tensor(float(done), dtype=torch.float32))
            value_list.append(output.value.squeeze(0))
            state = next_state

        rewards = torch.stack(reward_list).to(self.device)
        dones = torch.stack(done_list).to(self.device)
        values = torch.stack(value_list).to(self.device)
        advantages, returns = self.compute_gae(rewards, dones, values)

        diagnostics = {
            "mean_gross_return": float(np.mean(gross_returns)) if gross_returns else 0.0,
            "mean_net_return": float(np.mean(net_returns)) if net_returns else 0.0,
            "mean_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
            "mean_drawdown": float(np.mean(drawdowns)) if drawdowns else 0.0,
            "mean_variance": float(np.mean(variances)) if variances else 0.0,
            "mean_cost": float(np.mean(costs)) if costs else 0.0,
            "mean_reward_log_term": float(np.mean(reward_log_terms)) if reward_log_terms else 0.0,
            "mean_reward_return_bonus": float(np.mean(reward_return_bonus_terms)) if reward_return_bonus_terms else 0.0,
            "mean_penalty_turnover": float(np.mean(penalty_turnovers)) if penalty_turnovers else 0.0,
            "mean_penalty_drawdown": float(np.mean(penalty_drawdowns)) if penalty_drawdowns else 0.0,
            "mean_penalty_variance": float(np.mean(penalty_variances)) if penalty_variances else 0.0,
            "mean_reward_sharpe_bonus": float(np.mean(reward_sharpe_bonus_terms)) if reward_sharpe_bonus_terms else 0.0,
            "mean_reward_momentum_bonus": float(np.mean(reward_momentum_bonus_terms)) if reward_momentum_bonus_terms else 0.0,
        }

        return RolloutBatch(
            market=torch.stack(market_list).to(self.device),
            portfolio=torch.stack(portfolio_list).to(self.device),
            actions=torch.stack(action_list).to(self.device),
            log_probs=torch.stack(log_prob_list).to(self.device),
            rewards=rewards,
            dones=dones,
            values=values,
            returns=returns,
            advantages=advantages,
        ), raw_reward_sum, diagnostics

    def compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = float(self.cfg["gamma"])
        gae_lambda = float(self.cfg["gae_lambda"])
        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            last_advantage = delta + gamma * gae_lambda * mask * last_advantage
            advantages[t] = last_advantage
            next_value = values[t]
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update_policy(self, rollout: RolloutBatch, entropy_coef: float) -> dict[str, float]:
        clip_eps = float(self.cfg["clip_epsilon"])
        value_coef = float(self.cfg["value_loss_coef"])
        max_grad_norm = float(self.cfg.get("max_grad_norm", 1.0))
        ppo_epochs = int(self.cfg["ppo_epochs"])
        batch_size = rollout.market.shape[0]
        minibatch_size = min(int(self.cfg["minibatch_size"]), batch_size)

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []

        for _ in range(ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                market = rollout.market[mb_idx]
                portfolio = rollout.portfolio[mb_idx]
                actions = rollout.actions[mb_idx]
                old_log_probs = rollout.log_probs[mb_idx]
                returns = rollout.returns[mb_idx]
                advantages = rollout.advantages[mb_idx]

                with torch.amp.autocast(self.amp_device_type, enabled=self.scaler.is_enabled()):
                    new_log_probs, entropy, values = self.model.evaluate_actions(market, portfolio, actions)
                    ratio = (new_log_probs - old_log_probs).exp()
                    unclipped = ratio * advantages
                    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                    actor_loss = -torch.min(unclipped, clipped).mean()
                    critic_loss = nn.functional.mse_loss(values, returns)
                    entropy_loss = entropy.mean()
                    loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy_loss.item()))

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def evaluate_env(self, env: PortfolioEnv) -> dict[str, float]:
        trajectory = self.run_policy(env, deterministic=True)
        return compute_performance_metrics(
            portfolio_values=np.array(trajectory["portfolio_values"], dtype=np.float64),
            portfolio_returns=np.array(trajectory["portfolio_returns"], dtype=np.float64),
            turnover=np.array(trajectory["turnover"], dtype=np.float64),
            weights=np.array(trajectory["weights"], dtype=np.float64),
            trading_days=252,
        )

    def run_policy(self, env: PortfolioEnv, deterministic: bool = True) -> dict[str, Any]:
        state = env.reset()
        done = False
        outputs = {
            "dates": [],
            "portfolio_values": [env.portfolio_value],
            "portfolio_returns": [],
            "turnover": [],
            "weights": [env.current_weights.copy()],
            "rewards": [],
        }
        while not done:
            market = self._tensor(state["market"]).unsqueeze(0)
            portfolio = self._tensor(state["portfolio"]).unsqueeze(0)
            with torch.no_grad():
                policy_output = self.model(market, portfolio, deterministic=deterministic)
            state, reward, done, info = env.step(policy_output.action.squeeze(0).cpu().numpy())
            outputs["dates"].append(info["date"])
            outputs["portfolio_values"].append(info["portfolio_value"])
            outputs["portfolio_returns"].append(info["net_return"])
            outputs["turnover"].append(info["turnover"])
            outputs["weights"].append(info["weights"])
            outputs["rewards"].append(reward)
        return outputs

    def save_checkpoint(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_score": self.best_score,
            "reward_stats": {
                "count": self.reward_stats.count,
                "mean": self.reward_stats.mean,
                "m2": self.reward_stats.m2,
            },
            "config": self.cfg,
        }
        latest_path = self.checkpoint_path.parent / "latest_model.pt"
        torch.save(state, latest_path)
        if is_best:
            torch.save(state, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_score = float(checkpoint.get("best_score", -np.inf))
        reward_stats = checkpoint.get("reward_stats", {})
        self.reward_stats.count = int(reward_stats.get("count", 0))
        self.reward_stats.mean = float(reward_stats.get("mean", 0.0))
        self.reward_stats.m2 = float(reward_stats.get("m2", 0.0))
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        LOGGER.info("Resumed training from epoch %s", self.start_epoch)

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32, device=self.device)

    def _cosine_decay(self, step: int, total_steps: int, start: float, end: float) -> float:
        if total_steps <= 1:
            return end
        progress = min(max(step / (total_steps - 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(end + (start - end) * cosine)
