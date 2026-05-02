"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.metrics import compute_drawdown


def _prepare_output(path: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_training_curves(history: dict[str, list[float]], output_dir: str) -> None:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history.get("train_reward", []), label="Train Reward")
    plt.plot(history.get("val_sharpe", []), label="Val Sharpe")
    plt.legend()
    plt.title("Training Reward Curve")
    plt.tight_layout()
    plt.savefig(output_dir_path / "training_reward_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.get("actor_loss", []), label="Actor Loss")
    plt.plot(history.get("critic_loss", []), label="Critic Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(output_dir_path / "loss_curve.png", dpi=200)
    plt.close()


def plot_equity_curve(dates: list[str], values: np.ndarray, output_path: str, title: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(dates, values[1:], label="Portfolio Value")
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(_prepare_output(output_path), dpi=200)
    plt.close()


def plot_drawdown(dates: list[str], values: np.ndarray, output_path: str) -> None:
    dd = compute_drawdown(values[1:])
    plt.figure(figsize=(12, 4))
    plt.fill_between(dates, dd, 0.0, color="salmon", alpha=0.7)
    plt.xticks(rotation=45)
    plt.title("Drawdown")
    plt.tight_layout()
    plt.savefig(_prepare_output(output_path), dpi=200)
    plt.close()


def plot_weights_heatmap(dates: list[str], weights: np.ndarray, tickers: list[str], output_path: str) -> None:
    plt.figure(figsize=(12, 6))
    sns.heatmap(pd.DataFrame(weights[1:, :-1], index=dates, columns=tickers).T, cmap="viridis")
    plt.title("Weight Allocation Heatmap")
    plt.tight_layout()
    plt.savefig(_prepare_output(output_path), dpi=200)
    plt.close()


def plot_baseline_comparison(comparison_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(12, 6))
    comparison_df[["Sharpe Ratio", "Total Return %", "Calmar Ratio"]].plot(kind="bar", figsize=(12, 6))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(_prepare_output(output_path), dpi=200)
    plt.close()


def plot_rolling_sharpe(
    dates: list[str],
    returns: np.ndarray,
    window: int,
    output_path: str,
    trading_days: int = 252,
) -> None:
    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std().replace(0.0, np.nan)
    rolling_sharpe = np.sqrt(trading_days) * rolling_mean / rolling_std
    plt.figure(figsize=(12, 4))
    plt.plot(dates, rolling_sharpe)
    plt.xticks(rotation=45)
    plt.title("Rolling Sharpe")
    plt.tight_layout()
    plt.savefig(_prepare_output(output_path), dpi=200)
    plt.close()
