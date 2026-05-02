"""Rolling long-only mean-variance baseline."""

from __future__ import annotations

import numpy as np


def _solve_long_only_markowitz(expected_returns: np.ndarray, covariance: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    covariance = covariance + np.eye(covariance.shape[0]) * ridge
    raw = np.linalg.pinv(covariance) @ expected_returns
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        return np.full_like(raw, 1.0 / len(raw))
    return raw / raw.sum()


def run_markowitz(
    returns: np.ndarray,
    initial_value: float,
    lookback: int = 60,
    rebalance_every: int = 21,
) -> dict[str, np.ndarray]:
    n_steps, n_assets = returns.shape
    weights = np.zeros((n_steps, n_assets), dtype=np.float64)
    current_weight = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
    for t in range(n_steps):
        if t >= lookback and t % rebalance_every == 0:
            window = returns[t - lookback : t]
            mu = window.mean(axis=0)
            cov = np.cov(window.T)
            current_weight = _solve_long_only_markowitz(mu, cov)
        weights[t] = current_weight
    port_returns = (weights * returns).sum(axis=1)
    values = initial_value * np.cumprod(1.0 + port_returns)
    turnover = np.zeros(n_steps, dtype=np.float64)
    turnover[1:] = np.abs(weights[1:] - weights[:-1]).sum(axis=1)
    return {
        "weights": weights,
        "portfolio_returns": port_returns,
        "portfolio_values": np.concatenate([[initial_value], values]),
        "turnover": turnover,
    }
