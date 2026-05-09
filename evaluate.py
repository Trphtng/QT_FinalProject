"""Evaluation entry point."""

from __future__ import annotations

import argparse
import json
import re
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
from src.feature_engineering import DataBundle, time_series_split, walk_forward_splits
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

LEGACY_FEATURE_NAMES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return",
    "LogReturn",
    "ATR14",
    "RSI14",
    "MACD",
    "MACDSignal",
    "MACDHist",
    "BBUpper",
    "BBMiddle",
    "BBLower",
    "SMA10",
    "SMA20",
    "EMA20",
    "Volatility20",
    "RealizedVol20",
    "Momentum20",
    "RollingCorrMarket20",
    "DrawdownLocalPeak",
]


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
        lambda_sharpe_bonus=float(cfg["environment"].get("lambda_sharpe_bonus", 0.0)),
        sharpe_window=int(cfg["environment"].get("sharpe_window", 20)),
        lambda_momentum_bonus=float(cfg["environment"].get("lambda_momentum_bonus", 0.0)),
        momentum_window=int(cfg["environment"].get("momentum_window", 20)),
        momentum_scale=float(cfg["environment"].get("momentum_scale", 50.0)),
        return_target=float(cfg["environment"].get("return_target", 0.0)),
        reward_mode=str(cfg["environment"]["reward_mode"]),
        risk_free_rate=float(cfg["environment"]["risk_free_rate"]),
        rebalance_frequency=int(cfg["environment"].get("rebalance_frequency", cfg["environment"].get("rebalance_every", 1))),
        rebalance_alpha=float(cfg["environment"].get("rebalance_alpha", 1.0)),
        include_prev_weights=bool(cfg["environment"].get("include_prev_weights", True)),
        start_index=int(split[0]),
        end_index=int(split[1]),
    )


def _infer_checkpoint_dims(checkpoint: dict, n_assets: int) -> tuple[int, int]:
    model_state = checkpoint["model_state_dict"]
    market_in = int(model_state["encoder.input_proj.weight"].shape[1])
    portfolio_dim = int(model_state["portfolio_proj.0.weight"].shape[1])
    if market_in % n_assets != 0:
        raise RuntimeError(f"Checkpoint market input dim {market_in} is not divisible by n_assets={n_assets}")
    expected_n_features = market_in // n_assets
    return expected_n_features, portfolio_dim


def _adapt_bundle_features(bundle: DataBundle, expected_n_features: int) -> DataBundle:
    if len(bundle.feature_names) == expected_n_features:
        return bundle
    if expected_n_features == len(LEGACY_FEATURE_NAMES):
        name_to_idx = {name: idx for idx, name in enumerate(bundle.feature_names)}
        if all(name in name_to_idx for name in LEGACY_FEATURE_NAMES):
            indices = [name_to_idx[name] for name in LEGACY_FEATURE_NAMES]
            return DataBundle(
                dates=bundle.dates,
                tickers=bundle.tickers,
                feature_names=LEGACY_FEATURE_NAMES.copy(),
                features=bundle.features[:, :, indices],
                returns=bundle.returns,
                covariances=bundle.covariances,
                prices=bundle.prices,
                market_regime=bundle.market_regime,
            )
    raise RuntimeError(
        f"Feature mismatch: bundle has {len(bundle.feature_names)} features but checkpoint expects {expected_n_features}. "
        "Please retrain with current config or provide a matching processed dataset/checkpoint pair."
    )


def _extract_fold_number(path: Path, base_checkpoint: Path) -> int | None:
    pattern = rf"^{re.escape(base_checkpoint.stem)}_fold(\d+){re.escape(base_checkpoint.suffix)}$"
    match = re.match(pattern, path.name)
    if not match:
        return None
    return int(match.group(1))


def _resolve_walk_forward_checkpoint(cfg: dict, bundle: DataBundle) -> tuple[Path, dict[str, tuple[int, int]]]:
    base_checkpoint = Path(cfg["training"]["checkpoint_path"])
    default_splits = time_series_split(
        n_steps=len(bundle.dates),
        train_ratio=float(cfg["data"]["train_ratio"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        test_ratio=float(cfg["data"]["test_ratio"]),
    )
    wf_splits = walk_forward_splits(len(bundle.dates), cfg["data"])
    walk_forward_enabled = bool(cfg["data"].get("walk_forward", {}).get("enabled", False) and wf_splits)
    if not walk_forward_enabled:
        return base_checkpoint, default_splits

    summary_path = Path(cfg["evaluation"].get("walk_forward_summary_path", "outputs/reports/walk_forward_summary.json"))
    selected_fold: int | None = None

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary:
            best_entry = max(summary, key=lambda item: float(item.get("score", float("-inf"))))
            selected_fold = int(best_entry["fold"])
            LOGGER.info("Using best walk-forward fold %s from %s", selected_fold, summary_path)

    if selected_fold is None:
        fold_candidates = sorted(base_checkpoint.parent.glob(f"{base_checkpoint.stem}_fold*{base_checkpoint.suffix}"))
        fold_candidates = [path for path in fold_candidates if _extract_fold_number(path, base_checkpoint) is not None]
        if not fold_candidates:
            LOGGER.warning("Walk-forward is enabled but no fold checkpoint was found. Falling back to %s", base_checkpoint)
            return base_checkpoint, default_splits
        latest_checkpoint = max(fold_candidates, key=lambda path: path.stat().st_mtime)
        selected_fold = _extract_fold_number(latest_checkpoint, base_checkpoint)
        LOGGER.info("Walk-forward summary not found. Using latest trained fold checkpoint %s", latest_checkpoint)

    if selected_fold is None or selected_fold < 1 or selected_fold > len(wf_splits):
        LOGGER.warning("Resolved fold %s is invalid for current dataset. Falling back to %s", selected_fold, base_checkpoint)
        return base_checkpoint, default_splits

    checkpoint_path = base_checkpoint.parent / f"{base_checkpoint.stem}_fold{selected_fold}{base_checkpoint.suffix}"
    if not checkpoint_path.exists():
        LOGGER.warning("Resolved fold checkpoint %s does not exist. Falling back to %s", checkpoint_path, base_checkpoint)
        return base_checkpoint, default_splits

    return checkpoint_path, wf_splits[selected_fold - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DRL portfolio model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the YAML config file.")
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help="Evaluate only the PPO model without comparing against baseline strategies.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specify which walk-forward fold to evaluate (e.g. --fold 7). Overrides auto-selection.",
    )
    parser.add_argument(
        "--list-folds",
        action="store_true",
        help="List all available fold checkpoints and their validation scores, then exit.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["training"])
    bundle = DataBundle.load(cfg["data"]["processed_dir"])

    # --list-folds: show available checkpoints and exit
    if args.list_folds:
        base_ckpt = Path(cfg["training"]["checkpoint_path"])
        summary_path = Path(cfg["evaluation"].get("walk_forward_summary_path", "outputs/reports/walk_forward_summary.json"))
        print(f"\n{'='*65}")
        print(f"  Available fold checkpoints")
        print(f"{'='*65}")
        scores: dict[int, float] = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for entry in summary:
                fold_num = int(entry["fold"])
                score = float(entry.get("score", 0.0))
                if score < 9000:   # ignore forced-selection sentinel
                    scores[fold_num] = score
        fold_candidates = sorted(base_ckpt.parent.glob(f"{base_ckpt.stem}_fold*{base_ckpt.suffix}"))
        for ckpt in fold_candidates:
            match = _extract_fold_number(ckpt, base_ckpt)
            if match is not None:
                val_score = scores.get(match, float("nan"))
                size_mb = ckpt.stat().st_size / 1e6
                print(f"  Fold {match:3d} | {ckpt.name}  ({size_mb:.1f} MB) | val_score={val_score:+.4f}")
        print(f"\nUsage: python evaluate.py --fold <N>")
        return

    # --fold: manually override fold selection
    if args.fold is not None:
        base_ckpt = Path(cfg["training"]["checkpoint_path"])
        wf_splits = walk_forward_splits(len(bundle.dates), cfg["data"])
        forced_fold = int(args.fold)
        if not wf_splits or forced_fold < 1 or forced_fold > len(wf_splits):
            LOGGER.error("--fold %d is out of range. Valid range: 1 to %d", forced_fold, len(wf_splits) if wf_splits else 0)
            return
        checkpoint_path = base_ckpt.parent / f"{base_ckpt.stem}_fold{forced_fold}{base_ckpt.suffix}"
        if not checkpoint_path.exists():
            LOGGER.error("Checkpoint for fold %d not found: %s", forced_fold, checkpoint_path)
            return
        selected_split = wf_splits[forced_fold - 1]
        LOGGER.info("Manually selected fold %d: %s", forced_fold, checkpoint_path)
    else:
        checkpoint_path, selected_split = _resolve_walk_forward_checkpoint(cfg, bundle)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    expected_n_features, expected_portfolio_dim = _infer_checkpoint_dims(checkpoint, len(bundle.tickers))
    bundle = _adapt_bundle_features(bundle, expected_n_features)
    include_prev_weights = expected_portfolio_dim > (len(bundle.tickers) + 1)
    cfg = dict(cfg)
    cfg["environment"] = dict(cfg["environment"])
    cfg["environment"]["include_prev_weights"] = include_prev_weights
    test_env = build_env(bundle, selected_split["test"], cfg)

    model = ActorCriticNetwork(
        n_assets=len(bundle.tickers),
        n_features=len(bundle.feature_names),
        portfolio_dim=expected_portfolio_dim,
        model_cfg=cfg["model"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
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

    report_payload = {"PPO Actor-Critic": rl_metrics}
    report_path = Path(cfg["evaluation"]["report_path"])
    comparison_df: pd.DataFrame | None = None

    if not args.rl_only:
        test_start, test_end = selected_split["test"]
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

        report_payload = {
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

        comparison_df = pd.DataFrame(report_payload).T
        comparison_csv = Path(cfg["evaluation"]["comparison_csv"])
        comparison_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(comparison_csv)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    figure_dir = Path(cfg["evaluation"].get("figure_dir", "outputs/figures"))
    figure_dir.mkdir(parents=True, exist_ok=True)

    eval_dates = list(rl_output["dates"])
    plot_equity_curve(
        eval_dates,
        np.array(rl_output["portfolio_values"]),
        str(figure_dir / "equity_curve_test.png"),
        "Test Equity Curve",
    )
    plot_drawdown(eval_dates, np.array(rl_output["portfolio_values"]), str(figure_dir / "drawdown_chart.png"))
    plot_weights_heatmap(
        eval_dates,
        np.array(rl_output["weights"]),
        bundle.tickers,
        str(figure_dir / "weight_heatmap.png"),
    )
    plot_rolling_sharpe(
        eval_dates,
        np.array(rl_output["portfolio_returns"]),
        int(cfg["evaluation"]["rolling_sharpe_window"]),
        str(figure_dir / "rolling_sharpe.png"),
    )

    LOGGER.info("Evaluation completed using checkpoint %s", checkpoint_path)
    LOGGER.info("Metrics saved to %s", report_path)
    LOGGER.info("\n%s", pd.DataFrame(report_payload).T.round(4).to_string())


if __name__ == "__main__":
    main()
