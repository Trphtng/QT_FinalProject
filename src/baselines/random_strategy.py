"""Random allocation baseline."""

from __future__ import annotations

import numpy as np


def run_random_allocation(
    returns: np.ndarray,
    initial_value: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_steps, n_assets = returns.shape
    weights = rng.dirichlet(np.ones(n_assets), size=n_steps)
    port_returns = (returns * weights).sum(axis=1)
    values = initial_value * np.cumprod(1.0 + port_returns)
    turnover = np.zeros(n_steps, dtype=np.float64)
    turnover[1:] = np.abs(weights[1:] - weights[:-1]).sum(axis=1)
    return {
        "weights": weights,
        "portfolio_returns": port_returns,
        "portfolio_values": np.concatenate([[initial_value], values]),
        "turnover": turnover,
    }
