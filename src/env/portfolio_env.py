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


class RollingMoments:
    """O(1) rolling mean/std tracking for reward shaping."""

    def __init__(self, maxlen: int) -> None:
        self.maxlen = max(1, int(maxlen))
        self.values: deque[float] = deque(maxlen=self.maxlen)
        self.sum = 0.0
        self.sum_sq = 0.0

    def reset(self) -> None:
        self.values.clear()
        self.sum = 0.0
        self.sum_sq = 0.0

    def append(self, value: float) -> None:
        if len(self.values) == self.maxlen:
            removed = self.values.popleft()
            self.sum -= removed
            self.sum_sq -= removed * removed
        self.values.append(value)
        self.sum += value
        self.sum_sq += value * value

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return self.sum / len(self.values)

    def std(self) -> float:
        if len(self.values) < 2:
            return 1e-8
        n = len(self.values)
        variance = max(self.sum_sq / n - (self.sum / n) ** 2, 0.0)
        return float(np.sqrt(max(variance, 1e-8)))

    def projected_mean_std(self, value: float) -> tuple[float, float]:
        count = len(self.values) + 1
        total = self.sum + value
        total_sq = self.sum_sq + value * value
        mean = total / count
        variance = max(total_sq / count - mean * mean, 0.0)
        return mean, float(np.sqrt(max(variance, 1e-8)))


class PortfolioEnv:
    """Multi-asset portfolio environment with improved reward function."""

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
        include_prev_weights: bool = True,
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
        self.include_prev_weights = bool(include_prev_weights)

        self.start_index = max(lookback_window, start_index or lookback_window)
        self.end_index = min(len(dates) - 1, end_index or (len(dates) - 1))

        self.n_assets = len(tickers)
        self.action_dim = self.n_assets + 1
        self.current_step = self.start_index
        self.max_episode_steps = self.end_index - self.start_index
        self.portfolio_value = initial_cash
        self.peak_value = initial_cash
        self.current_weights = np.zeros(self.action_dim, dtype=np.float32)
        self.prev_weights = self.current_weights.copy()
        self.current_weights[-1] = 1.0

        # Portfolio state: weights + prev_weights + 4 extra context signals
        # [rolling_vol, rolling_sharpe, current_drawdown, trend_signal]
        _base_dim = self.action_dim * 2 if self.include_prev_weights else self.action_dim
        self.portfolio_state_dim = _base_dim + 4

        self.recent_net_returns = RollingMoments(self.sharpe_window)
        self.rolling_vol_window = RollingMoments(self.sharpe_window)

        # EMA for smooth Sharpe bonus (alpha = 2 / (window+1))
        self._ema_alpha = 2.0 / (self.sharpe_window + 1)
        self._ema_return = 0.0
        self._ema_sq_return = 0.0
        self._ema_initialized = False

        self.momentum_return_cumsum = np.concatenate(
            [np.zeros((1, self.n_assets), dtype=np.float32), np.cumsum(self.returns, axis=0, dtype=np.float32)],
            axis=0,
        )
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
        self.recent_net_returns.reset()
        self.rolling_vol_window.reset()
        self._ema_return = 0.0
        self._ema_sq_return = 0.0
        self._ema_initialized = False
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

        # Update rolling trackers
        self.recent_net_returns.append(net_return)
        self.rolling_vol_window.append(net_return)
        self._update_ema(net_return)

        reward, reward_terms = self._compute_reward(
            asset_weights=asset_weights,
            net_return=net_return,
            turnover=turnover,
            variance=variance,
            drawdown=drawdown,
        )

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

    def _update_ema(self, net_return: float) -> None:
        """Update EMA statistics for smooth Sharpe bonus computation."""
        a = self._ema_alpha
        if not self._ema_initialized:
            self._ema_return = net_return
            self._ema_sq_return = net_return * net_return
            self._ema_initialized = True
        else:
            self._ema_return = a * net_return + (1 - a) * self._ema_return
            self._ema_sq_return = a * net_return * net_return + (1 - a) * self._ema_sq_return

    def _get_ema_sharpe(self) -> float:
        """Compute annualized Sharpe ratio from EMA statistics."""
        if not self._ema_initialized:
            return 0.0
        ema_var = max(self._ema_sq_return - self._ema_return ** 2, 1e-10)
        ema_std = float(np.sqrt(ema_var))
        sharpe = self._ema_return / max(ema_std, 1e-8)
        # Annualize (sqrt(252) daily → annualized)
        return float(np.clip(sharpe * np.sqrt(252), -5.0, 5.0))

    def _compute_reward(
        self,
        asset_weights: np.ndarray,
        net_return: float,
        turnover: float,
        variance: float,
        drawdown: float,
    ) -> tuple[float, dict[str, float]]:
        # ── BASE: log(1+r) — proper for portfolio compounding ──────────────────
        log_return = float(np.log1p(np.clip(net_return, -0.95, None)))

        # ── PENALTIES ──────────────────────────────────────────────────────────
        # Variance penalty: downside only (don't penalize upside vol)
        is_downside = 1.0 if net_return < 0.0 else 0.0
        penalty_variance = float(self.lambda_var * variance * is_downside)

        # Turnover penalty: small but nonzero to discourage noise trading
        penalty_turnover = float(self.lambda_turnover * max(turnover, 0.0))

        # Drawdown penalty: activates above threshold with power scaling
        dd_excess = max(drawdown - self.drawdown_penalty_threshold, 0.0)
        penalty_drawdown = float(self.lambda_drawdown * (dd_excess ** self.drawdown_penalty_power))

        # ── BONUSES ────────────────────────────────────────────────────────────
        # Return bonus: asymmetric — reward only positive excess return
        reward_return_bonus = float(self.lambda_return_bonus * max(net_return - self.return_target, 0.0))

        # Sharpe bonus: EMA-based (smooth, lag-free, annualized)
        reward_sharpe_bonus = self._compute_sharpe_bonus_ema()

        # Momentum bonus: rewards aligning weights with recent asset trends
        reward_momentum_bonus = self._compute_momentum_bonus(asset_weights)

        # ── COMBINE & CLIP ─────────────────────────────────────────────────────
        reward = (
            log_return
            + reward_return_bonus
            + reward_sharpe_bonus
            + reward_momentum_bonus
            - penalty_variance
            - penalty_turnover
            - penalty_drawdown
        )
        # Clip to prevent extreme values from dominating gradient
        reward = float(np.clip(np.nan_to_num(reward, nan=-0.1, posinf=0.1, neginf=-0.1), -0.1, 0.1))

        reward_terms = {
            "reward_log_term": log_return,
            "reward_return_bonus": reward_return_bonus,
            "reward_sharpe_bonus": reward_sharpe_bonus,
            "reward_momentum_bonus": reward_momentum_bonus,
            "penalty_variance": penalty_variance,
            "penalty_turnover": penalty_turnover,
            "penalty_drawdown": penalty_drawdown,
        }
        return reward, reward_terms

    def _compute_sharpe_bonus_ema(self) -> float:
        """Smooth EMA-based Sharpe bonus — avoids the noise of rolling window."""
        if self.lambda_sharpe_bonus <= 0.0 or not self._ema_initialized:
            return 0.0
        annualized_sharpe = self._get_ema_sharpe()
        # tanh maps to [-1, 1]; divide by 3 so tanh saturates near Sharpe=3
        return float(self.lambda_sharpe_bonus * np.tanh(annualized_sharpe / 3.0))

    def _compute_sharpe_bonus(self, net_return: float) -> float:
        """Legacy rolling-window Sharpe bonus (kept for compatibility)."""
        if self.lambda_sharpe_bonus <= 0.0:
            return 0.0
        if len(self.recent_net_returns.values) + 1 < 5:
            return 0.0
        mean, std = self.recent_net_returns.projected_mean_std(net_return)
        sharpe = mean / max(std, 1e-8)
        return float(self.lambda_sharpe_bonus * np.tanh(sharpe))

    def _compute_momentum_bonus(self, asset_weights: np.ndarray) -> float:
        """Reward aligning portfolio weights with recent asset momentum."""
        if self.lambda_momentum_bonus <= 0.0:
            return 0.0
        start = max(self.start_index, self.current_step - self.momentum_window + 1)
        end = self.current_step + 1
        window_len = end - start
        if window_len < 3:
            return 0.0
        trailing_sum = self.momentum_return_cumsum[end] - self.momentum_return_cumsum[start]
        trailing_mean = trailing_sum / float(window_len)
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

        # Base portfolio state: current + previous weights
        if self.include_prev_weights:
            base_state = np.concatenate([self.current_weights, self.prev_weights], axis=0)
        else:
            base_state = self.current_weights.copy()

        # ── Extra context signals (4 scalars) ─────────────────────────────────
        # 1. Rolling volatility (normalized to daily std scale, ~1%)
        rolling_vol = float(self.rolling_vol_window.std()) * 100.0  # scale to ~[0, 5]

        # 2. Rolling Sharpe (EMA-based, annualized, clipped)
        rolling_sharpe = float(np.clip(self._get_ema_sharpe() / 5.0, -1.0, 1.0))  # scale to [-1, 1]

        # 3. Current drawdown (already in [0, 1], negate so less is better)
        current_drawdown = float(1.0 - self.portfolio_value / max(self.peak_value, 1e-8))

        # 4. Trend signal: recent mean market return across assets (scaled)
        window = min(5, max(1, self.current_step - self.start_index))
        trend_start = max(0, self.current_step - window)
        trend_signal = float(np.mean(self.returns[trend_start:self.current_step])) * 100.0

        extra_signals = np.array(
            [rolling_vol, rolling_sharpe, current_drawdown, trend_signal],
            dtype=np.float32,
        )
        extra_signals = np.nan_to_num(extra_signals, nan=0.0, posinf=1.0, neginf=-1.0)

        portfolio_state = np.concatenate([base_state, extra_signals], axis=0)
        return {
            "market": market_window,
            "portfolio": portfolio_state,
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
