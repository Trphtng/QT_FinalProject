"""Equal-weight buy and hold baseline."""

from __future__ import annotations

import numpy as np


def run_buy_hold_equal_weight(returns: np.ndarray, initial_value: float) -> dict[str, np.ndarray]:
    n_assets = returns.shape[1]
    weights = np.full((returns.shape[0], n_assets), 1.0 / n_assets, dtype=np.float64)
    port_returns = (returns * weights).sum(axis=1)
    values = initial_value * np.cumprod(1.0 + port_returns)
    return {
        "weights": weights,
        "portfolio_returns": port_returns,
        "portfolio_values": np.concatenate([[initial_value], values]),
        "turnover": np.zeros_like(port_returns),
    }
