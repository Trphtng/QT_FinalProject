"""Evaluation entry point."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.agents.trainer import PPOTrainer
from src.baselines.buy_hold import run_buy_hold_equal_weight
from src.baselines.markowitz import run_markowitz
from src.baselines.random_strategy import run_random_allocation
from src.env.portfolio_env import PortfolioEnv
from src.feature_engineering import DataBundle, time_series_split
from src.models.actor_critic import ActorCriticNetwork
from src.utils.logger import get_logger
from src.utils.metrics import compute_performance_metrics
from src.utils.plotting import (
    plot_baseline_comparison,
    plot_drawdown,
    plot_equity_curve,
    plot_rolling_sharpe,
    plot_weights_heatmap,
)
from train import get_device, load_config


LOGGER = get_logger(__name__)


def build_env(bundle, split, cfg) -> PortfolioEnv:
    return PortfolioEnv(
        features=bundle.features,
        returns=bundle.returns,
        covariances=bundle.covariances,
        dates=bundle.dates,
        tickers=bundle.tickers,
        lookback_window=int(cfg["data"]["lookback_window"]),
        initial_cash=float(cfg["environment"]["initial_cash"]),
        fee_rate=float(cfg["environment"]["fee_rate"]),
        slippage_rate=float(cfg["environment"]["slippage_rate"]),
        kappa=float(cfg["environment"]["kappa"]),
        lambda_var=float(cfg["environment"]["lambda_var"]),
        lambda_turnover=float(cfg["environment"].get("lambda_turnover", 0.0)),
        lambda_drawdown=float(cfg["environment"].get("lambda_drawdown", 0.0)),
        drawdown_penalty_threshold=float(cfg["environment"].get("drawdown_penalty_threshold", 0.08)),
        drawdown_penalty_power=float(cfg["environment"].get("drawdown_penalty_power", 1.5)),
        lambda_return_bonus=float(cfg["environment"].get("lambda_return_bonus", 0.0)),
        return_target=float(cfg["environment"].get("return_target", 0.0)),
        reward_mode=str(cfg["environment"]["reward_mode"]),
        risk_free_rate=float(cfg["environment"]["risk_free_rate"]),
        rebalance_frequency=int(cfg["environment"].get("rebalance_frequency", cfg["environment"].get("rebalance_every", 1))),
        rebalance_alpha=float(cfg["environment"].get("rebalance_alpha", 1.0)),
        start_index=int(split[0]),
        end_index=int(split[1]),
    )


def main() -> None:
    cfg = load_config()
    device = get_device(cfg["training"])
    bundle = DataBundle.load(cfg["data"]["processed_dir"])
    splits = time_series_split(
        n_steps=len(bundle.dates),
        train_ratio=float(cfg["data"]["train_ratio"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        test_ratio=float(cfg["data"]["test_ratio"]),
    )
    test_env = build_env(bundle, splits["test"], cfg)

    model = ActorCriticNetwork(
        n_assets=len(bundle.tickers),
        n_features=len(bundle.feature_names),
        portfolio_dim=test_env.portfolio_state_dim,
        model_cfg=cfg["model"],
    )
    checkpoint = torch.load(cfg["training"]["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    trainer = PPOTrainer(model=model, train_env=test_env, val_env=test_env, cfg=cfg["training"], device=device)
    rl_output = trainer.run_policy(test_env, deterministic=True)
    rl_metrics = compute_performance_metrics(
        portfolio_values=np.array(rl_output["portfolio_values"], dtype=np.float64),
        portfolio_returns=np.array(rl_output["portfolio_returns"], dtype=np.float64),
        turnover=np.array(rl_output["turnover"], dtype=np.float64),
        weights=np.array(rl_output["weights"], dtype=np.float64),
    )

    test_start, test_end = splits["test"]
    test_returns = bundle.returns[test_start:test_end]
    initial_cash = float(cfg["environment"]["initial_cash"])

    buy_hold_output = run_buy_hold_equal_weight(test_returns, initial_cash)
    markowitz_output = run_markowitz(
        test_returns,
        initial_value=initial_cash,
        lookback=max(20, int(cfg["data"]["lookback_window"]) * 2),
        rebalance_every=int(cfg["evaluation"]["benchmark_rebalance_every"]),
    )
    random_output = run_random_allocation(
        test_returns,
        initial_value=initial_cash,
        seed=int(cfg["project"]["seed"]),
    )

    comparison = {
        "PPO Actor-Critic": rl_metrics,
        "BuyHold EqualWeight": compute_performance_metrics(
            buy_hold_output["portfolio_values"],
            buy_hold_output["portfolio_returns"],
            buy_hold_output["turnover"],
            buy_hold_output["weights"],
        ),
        "Markowitz": compute_performance_metrics(
            markowitz_output["portfolio_values"],
            markowitz_output["portfolio_returns"],
            markowitz_output["turnover"],
            markowitz_output["weights"],
        ),
        "Random Allocation": compute_performance_metrics(
            random_output["portfolio_values"],
            random_output["portfolio_returns"],
            random_output["turnover"],
            random_output["weights"],
        ),
    }

    comparison_df = pd.DataFrame(comparison).T
    comparison_csv = Path(cfg["evaluation"]["comparison_csv"])
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_csv)

    report_path = Path(cfg["evaluation"]["report_path"])
    report_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    eval_dates = list(rl_output["dates"])
    plot_equity_curve(eval_dates, np.array(rl_output["portfolio_values"]), "outputs/figures/equity_curve_test.png", "Test Equity Curve")
    plot_drawdown(eval_dates, np.array(rl_output["portfolio_values"]), "outputs/figures/drawdown_chart.png")
    plot_weights_heatmap(eval_dates, np.array(rl_output["weights"]), bundle.tickers, "outputs/figures/weight_heatmap.png")
    plot_baseline_comparison(comparison_df, "outputs/figures/baseline_comparison.png")
    plot_rolling_sharpe(
        eval_dates,
        np.array(rl_output["portfolio_returns"]),
        int(cfg["evaluation"]["rolling_sharpe_window"]),
        "outputs/figures/rolling_sharpe.png",
    )

    LOGGER.info("Evaluation completed. Metrics saved to %s", report_path)
    LOGGER.info("\n%s", comparison_df.round(4).to_string())


if __name__ == "__main__":
    main()
