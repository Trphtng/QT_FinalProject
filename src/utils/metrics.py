"""Performance metrics for portfolio backtests."""

from __future__ import annotations

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-12 else 0.0


def compute_drawdown(values: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(values)
    return values / np.maximum(peaks, 1e-12) - 1.0




def compute_performance_metrics(
    portfolio_values: np.ndarray,
    portfolio_returns: np.ndarray,
    turnover: np.ndarray,
    weights: np.ndarray,
    trading_days: int = 252,
) -> dict[str, float]:
    total_return = float(portfolio_values[-1] / portfolio_values[0] - 1.0)
    n_periods = max(len(portfolio_returns), 1)
    years = n_periods / trading_days
    cagr = float((portfolio_values[-1] / portfolio_values[0]) ** (1 / max(years, 1e-8)) - 1.0)
    mean_return = float(np.mean(portfolio_returns))
    std_return = float(np.std(portfolio_returns))
    sharpe = _safe_div(np.sqrt(trading_days) * mean_return, std_return)
    drawdowns = compute_drawdown(portfolio_values)
    max_drawdown = float(np.min(drawdowns))
    volatility = float(std_return * np.sqrt(trading_days))
    calmar = _safe_div(cagr, abs(max_drawdown))
    win_rate = float((portfolio_returns > 0).mean())
    avg_turnover = float(np.mean(turnover)) if turnover.size else 0.0
    return {
        "Total Return %": total_return * 100.0,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Volatility": volatility,
        "Calmar Ratio": calmar,
        "Win Rate": win_rate,
        "Final Portfolio Value": float(portfolio_values[-1]),
        "Turnover": avg_turnover,
    }
