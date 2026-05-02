"""Training entry point."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.agents.trainer import PPOTrainer
from src.data_loader import YahooDataLoader, create_download_config
from src.env.portfolio_env import PortfolioEnv
from src.feature_engineering import engineer_features, time_series_split, walk_forward_splits
from src.models.actor_critic import ActorCriticNetwork
from src.utils.logger import get_logger
from src.utils.plotting import plot_training_curves
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_device(training_cfg: dict) -> torch.device:
    requested = training_cfg.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


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


def main() -> None:
    cfg = load_config()
    set_seed(int(cfg["project"]["seed"]))
    device = get_device(cfg["training"])
    LOGGER.info("Using device: %s", device)

    loader = YahooDataLoader(create_download_config(cfg["data"]))
    raw_df = loader.load()
    bundle = engineer_features(raw_df, cfg["features"], cfg["data"])
    bundle.save(cfg["data"]["processed_dir"])

    default_splits = time_series_split(
        n_steps=len(bundle.dates),
        train_ratio=float(cfg["data"]["train_ratio"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        test_ratio=float(cfg["data"]["test_ratio"]),
    )
    wf_splits = walk_forward_splits(len(bundle.dates), cfg["data"])
    use_walk_forward = bool(cfg["data"].get("walk_forward", {}).get("enabled", False) and wf_splits)
    splits_to_run = wf_splits if use_walk_forward else [default_splits]

    all_fold_metrics: list[dict[str, float]] = []
    final_history: dict[str, list[float]] = {}
    for fold_idx, split in enumerate(splits_to_run):
        train_env = build_env(bundle, split["train"], cfg)
        val_env = build_env(bundle, split["val"], cfg)
        model = ActorCriticNetwork(
            n_assets=len(bundle.tickers),
            n_features=len(bundle.feature_names),
            portfolio_dim=train_env.portfolio_state_dim,
            model_cfg=cfg["model"],
        )

        fold_training_cfg = dict(cfg["training"])
        base_ckpt = Path(str(cfg["training"]["checkpoint_path"]))
        if use_walk_forward:
            fold_ckpt = base_ckpt.parent / f"{base_ckpt.stem}_fold{fold_idx + 1}{base_ckpt.suffix}"
            fold_training_cfg["checkpoint_path"] = str(fold_ckpt)
            fold_training_cfg["tensorboard_dir"] = f"{cfg['training']['tensorboard_dir']}/fold_{fold_idx + 1}"
            fold_training_cfg["resume_from"] = None

        trainer = PPOTrainer(model=model, train_env=train_env, val_env=val_env, cfg=fold_training_cfg, device=device)
        history = trainer.train()
        final_history = history
        val_metrics = trainer.evaluate_env(val_env)
        score = float(val_metrics["Sharpe Ratio"] - 0.5 * abs(val_metrics["Max Drawdown"]))
        all_fold_metrics.append(
            {
                "fold": float(fold_idx + 1),
                "score": score,
                "sharpe": float(val_metrics["Sharpe Ratio"]),
                "max_drawdown": float(val_metrics["Max Drawdown"]),
                "cagr": float(val_metrics["CAGR"]),
                "final_value": float(val_metrics["Final Portfolio Value"]),
            }
        )
        LOGGER.info("Fold %s completed | Score %.4f | Sharpe %.4f | MDD %.4f", fold_idx + 1, score, val_metrics["Sharpe Ratio"], val_metrics["Max Drawdown"])

    plot_training_curves(final_history, "outputs/figures")
    summary_path = Path("outputs/reports/train_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(final_history, indent=2), encoding="utf-8")

    if all_fold_metrics:
        fold_path = Path("outputs/reports/walk_forward_summary.json")
        fold_path.write_text(json.dumps(all_fold_metrics, indent=2), encoding="utf-8")
        mean_score = float(sum(metric["score"] for metric in all_fold_metrics) / len(all_fold_metrics))
        LOGGER.info("Training completed | Walk-forward mean score %.4f | Best model path %s", mean_score, cfg["training"]["checkpoint_path"])
    else:
        LOGGER.info("Training completed. Best model saved to %s", cfg["training"]["checkpoint_path"])


if __name__ == "__main__":
    main()
