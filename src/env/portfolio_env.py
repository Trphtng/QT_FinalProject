"""Portfolio management environment."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any

import numpy as np


@dataclass
class StepInfo:
    date: str
    portfolio_value: float
    peak_value: float
    turnover: float
    transaction_cost: float
    variance: float
    gross_return: float
    net_return: float
    drawdown: float
    reward: float
    reward_log_term: float
    reward_return_bonus: float
    penalty_variance: float
    penalty_turnover: float
    penalty_drawdown: float
    reward_sharpe_bonus: float
    reward_momentum_bonus: float
    cash_weight: float
    rebalanced: bool
    weights: np.ndarray


class PortfolioEnv:
    """A simple multi-asset portfolio environment with weight actions."""

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        covariances: np.ndarray,
        dates: list[str],
        tickers: list[str],
        lookback_window: int,
        initial_cash: float,
        fee_rate: float,
        slippage_rate: float,
        kappa: float,
        lambda_var: float,
        lambda_turnover: float = 0.0,
        lambda_drawdown: float = 0.0,
        drawdown_penalty_threshold: float = 0.08,
        drawdown_penalty_power: float = 1.5,
        lambda_return_bonus: float = 0.5,
        return_target: float = 0.0,
        lambda_sharpe_bonus: float = 0.15,
        sharpe_window: int = 20,
        lambda_momentum_bonus: float = 0.10,
        momentum_window: int = 20,
        momentum_scale: float = 50.0,
        reward_mode: str = "log_return",
        risk_free_rate: float = 0.0,
        rebalance_frequency: int = 1,
        rebalance_alpha: float = 1.0,
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> None:
        self.features = features
        self.returns = returns
        self.covariances = covariances
        self.dates = dates
        self.tickers = tickers
        self.lookback_window = lookback_window
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.kappa = kappa
        self.lambda_var = lambda_var
        self.lambda_turnover = lambda_turnover
        self.lambda_drawdown = lambda_drawdown
        self.drawdown_penalty_threshold = float(max(0.0, drawdown_penalty_threshold))
        self.drawdown_penalty_power = float(max(1.0, drawdown_penalty_power))
        self.lambda_return_bonus = float(max(0.0, lambda_return_bonus))
        self.return_target = float(return_target)
        self.lambda_sharpe_bonus = float(max(0.0, lambda_sharpe_bonus))
        self.sharpe_window = max(5, int(sharpe_window))
        self.lambda_momentum_bonus = float(max(0.0, lambda_momentum_bonus))
        self.momentum_window = max(3, int(momentum_window))
        self.momentum_scale = float(max(1e-6, momentum_scale))
        self.reward_mode = reward_mode
        self.risk_free_rate = risk_free_rate
        self.rebalance_frequency = max(1, int(rebalance_frequency))
        self.rebalance_alpha = float(np.clip(rebalance_alpha, 0.0, 1.0))

        self.start_index = max(lookback_window, start_index or lookback_window)
        self.end_index = min(len(dates) - 1, end_index or (len(dates) - 1))

        self.n_assets = len(tickers)
        self.action_dim = self.n_assets + 1
        self.current_step = self.start_index
        self.portfolio_value = initial_cash
        self.peak_value = initial_cash
        self.current_weights = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_weights = self.current_weights.copy()
        self.current_weights[-1] = 1.0
        self.portfolio_state_dim = self.action_dim * 2
        self.recent_net_returns: deque[float] = deque(maxlen=self.sharpe_window)
        self.turnover_history: list[float] = []
        self.weight_history: list[np.ndarray] = []
        self.value_history: list[float] = []
        self.reward_history: list[float] = []
        self.info_history: list[StepInfo] = []

    def reset(self) -> dict[str, np.ndarray]:
        self.current_step = self.start_index
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        self.current_weights = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_weights = self.current_weights.copy()
        self.current_weights[-1] = 1.0
        self.turnover_history = []
        self.weight_history = [self.current_weights.copy()]
        self.value_history = [self.portfolio_value]
        self.reward_history = []
        self.info_history = []
        self.recent_net_returns = deque(maxlen=self.sharpe_window)
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        prev_weights = self.current_weights.copy()
        self.prev_weights = prev_weights.copy()
        target_weights = self._normalize_action(action)
        rebalanced = self._should_rebalance()
        if rebalanced:
            weights = (1.0 - self.rebalance_alpha) * prev_weights + self.rebalance_alpha * target_weights
            weights = self._normalize_action(weights)
        else:
            weights = prev_weights.copy()

        asset_weights = weights[:-1]
        cash_weight = float(weights[-1])
        next_returns = self.returns[self.current_step]
        cov = self.covariances[self.current_step]

        turnover = float(np.abs(weights - prev_weights).sum())
        transaction_cost = float(turnover * (self.fee_rate + self.slippage_rate))
        gross_return = float(np.dot(asset_weights, next_returns) + cash_weight * self.risk_free_rate)
        variance = float(asset_weights.T @ cov @ asset_weights)
        net_return = gross_return - transaction_cost

        net_growth = max(1e-8, 1.0 + np.clip(net_return, -0.999999, None))
        self.portfolio_value *= net_growth
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = float(1.0 - self.portfolio_value / max(self.peak_value, 1e-8))
        reward, reward_terms = self._compute_reward(
            asset_weights=asset_weights,
            net_return=net_return,
            turnover=turnover,
            variance=variance,
            drawdown=drawdown,
        )
        self.recent_net_returns.append(net_return)

        self.current_weights = weights
        self.current_step += 1
        done = self.current_step >= self.end_index

        info = StepInfo(
            date=self.dates[self.current_step - 1],
            portfolio_value=self.portfolio_value,
            peak_value=self.peak_value,
            turnover=turnover,
            transaction_cost=transaction_cost,
            variance=variance,
            gross_return=gross_return,
            net_return=net_return,
            drawdown=drawdown,
            reward=reward,
            reward_log_term=reward_terms["reward_log_term"],
            reward_return_bonus=reward_terms["reward_return_bonus"],
            penalty_variance=reward_terms["penalty_variance"],
            penalty_turnover=reward_terms["penalty_turnover"],
            penalty_drawdown=reward_terms["penalty_drawdown"],
            reward_sharpe_bonus=reward_terms["reward_sharpe_bonus"],
            reward_momentum_bonus=reward_terms["reward_momentum_bonus"],
            cash_weight=cash_weight,
            rebalanced=rebalanced,
            weights=weights.copy(),
        )
        self.turnover_history.append(turnover)
        self.weight_history.append(weights.copy())
        self.value_history.append(self.portfolio_value)
        self.reward_history.append(reward)
        self.info_history.append(info)

        next_state = self._get_state()
        return next_state, reward, done, self._info_to_dict(info)

    def _compute_reward(
        self,
        asset_weights: np.ndarray,
        net_return: float,
        turnover: float,
        variance: float,
        drawdown: float,
    ) -> tuple[float, dict[str, float]]:
        safe_return = float(np.clip(net_return, -0.95, None))
        reward_log_term = float(np.log1p(safe_return))
        reward_return_bonus = float(self.lambda_return_bonus * max(net_return - self.return_target, 0.0))
        reward_sharpe_bonus = self._compute_sharpe_bonus(net_return)
        reward_momentum_bonus = self._compute_momentum_bonus(asset_weights)
        penalty_variance = float(self.lambda_var * max(variance, 0.0))
        penalty_turnover = float(self.lambda_turnover * max(turnover, 0.0))
        dd_excess = max(drawdown - self.drawdown_penalty_threshold, 0.0)
        penalty_drawdown = float(self.lambda_drawdown * (dd_excess ** self.drawdown_penalty_power))
        reward = (
            reward_log_term
            + reward_return_bonus
            + reward_sharpe_bonus
            + reward_momentum_bonus
            - penalty_variance
            - penalty_turnover
            - penalty_drawdown
        )
        reward = float(np.nan_to_num(reward, nan=-1.0, posinf=1.0, neginf=-1.0))
        reward_terms = {
            "reward_log_term": reward_log_term,
            "reward_return_bonus": reward_return_bonus,
            "reward_sharpe_bonus": reward_sharpe_bonus,
            "reward_momentum_bonus": reward_momentum_bonus,
            "penalty_variance": penalty_variance,
            "penalty_turnover": penalty_turnover,
            "penalty_drawdown": penalty_drawdown,
        }
        return reward, reward_terms

    def _compute_sharpe_bonus(self, net_return: float) -> float:
        if self.lambda_sharpe_bonus <= 0.0:
            return 0.0
        series = list(self.recent_net_returns)
        series.append(net_return)
        if len(series) < 5:
            return 0.0
        mean = float(np.mean(series))
        std = float(np.std(series))
        sharpe = mean / max(std, 1e-8)
        return float(self.lambda_sharpe_bonus * np.tanh(sharpe))

    def _compute_momentum_bonus(self, asset_weights: np.ndarray) -> float:
        if self.lambda_momentum_bonus <= 0.0:
            return 0.0
        start = max(self.start_index, self.current_step - self.momentum_window + 1)
        window = self.returns[start : self.current_step + 1]
        if window.shape[0] < 3:
            return 0.0
        trailing_mean = window.mean(axis=0)
        signal = float(np.dot(asset_weights, trailing_mean))
        return float(self.lambda_momentum_bonus * np.tanh(signal * self.momentum_scale))

    def _should_rebalance(self) -> bool:
        relative_step = self.current_step - self.start_index
        return relative_step % self.rebalance_frequency == 0

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.nan_to_num(action.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, 1e-8, None)
        total = action.sum()
        if total <= 0:
            weights = np.zeros_like(action)
            weights[-1] = 1.0
            return weights
        return action / total

    def _get_state(self) -> dict[str, np.ndarray]:
        start = self.current_step - self.lookback_window
        end = self.current_step
        market_window = self.features[start:end]
        portfolio_state = np.concatenate(
            [self.current_weights, self.prev_weights],
            axis=0,
        )
        return {
            "market": market_window.astype(np.float32),
            "portfolio": portfolio_state.astype(np.float32),
        }

    def _info_to_dict(self, info: StepInfo) -> dict[str, Any]:
        return {
            "date": info.date,
            "portfolio_value": info.portfolio_value,
            "peak_value": info.peak_value,
            "turnover": info.turnover,
            "transaction_cost": info.transaction_cost,
            "cost": info.transaction_cost,
            "variance": info.variance,
            "gross_return": info.gross_return,
            "net_return": info.net_return,
            "drawdown": info.drawdown,
            "reward": info.reward,
            "reward_log_term": info.reward_log_term,
            "reward_return_bonus": info.reward_return_bonus,
            "penalty_variance": info.penalty_variance,
            "penalty_turnover": info.penalty_turnover,
            "penalty_drawdown": info.penalty_drawdown,
            "reward_sharpe_bonus": info.reward_sharpe_bonus,
            "reward_momentum_bonus": info.reward_momentum_bonus,
            "cash_weight": info.cash_weight,
            "rebalanced": info.rebalanced,
            "weights": info.weights,
        }
