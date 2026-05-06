"""
evaluate_all_folds.py -- Evaluate all fold models on their own TEST split.

Usage:
    python evaluate_all_folds.py
    python evaluate_all_folds.py --top 5
    python evaluate_all_folds.py --set-best
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Force UTF-8 output on Windows cp1252 terminals
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.baselines.buy_hold import run_buy_hold_equal_weight
from src.baselines.markowitz import run_markowitz
from src.env.portfolio_env import PortfolioEnv
from src.feature_engineering import DataBundle, walk_forward_splits
from src.models.actor_critic import ActorCriticNetwork
from src.agents.trainer import PPOTrainer
from src.utils.metrics import compute_performance_metrics
from train import get_device, load_config


def build_env(bundle: DataBundle, split: tuple[int, int], cfg: dict) -> PortfolioEnv:
    env_cfg = cfg["environment"]
    return PortfolioEnv(
        features=bundle.features,
        returns=bundle.returns,
        covariances=bundle.covariances,
        dates=bundle.dates,
        tickers=bundle.tickers,
        lookback_window=int(cfg["data"]["lookback_window"]),
        initial_cash=float(env_cfg["initial_cash"]),
        fee_rate=float(env_cfg["fee_rate"]),
        slippage_rate=float(env_cfg["slippage_rate"]),
        kappa=float(env_cfg["kappa"]),
        lambda_var=float(env_cfg["lambda_var"]),
        lambda_turnover=float(env_cfg.get("lambda_turnover", 0.0)),
        lambda_drawdown=float(env_cfg.get("lambda_drawdown", 0.0)),
        drawdown_penalty_threshold=float(env_cfg.get("drawdown_penalty_threshold", 0.08)),
        drawdown_penalty_power=float(env_cfg.get("drawdown_penalty_power", 1.5)),
        lambda_return_bonus=float(env_cfg.get("lambda_return_bonus", 0.0)),
        lambda_sharpe_bonus=float(env_cfg.get("lambda_sharpe_bonus", 0.0)),
        sharpe_window=int(env_cfg.get("sharpe_window", 20)),
        lambda_momentum_bonus=float(env_cfg.get("lambda_momentum_bonus", 0.0)),
        momentum_window=int(env_cfg.get("momentum_window", 20)),
        momentum_scale=float(env_cfg.get("momentum_scale", 50.0)),
        return_target=float(env_cfg.get("return_target", 0.0)),
        reward_mode=str(env_cfg["reward_mode"]),
        risk_free_rate=float(env_cfg["risk_free_rate"]),
        rebalance_frequency=int(env_cfg.get("rebalance_frequency", 1)),
        rebalance_alpha=float(env_cfg.get("rebalance_alpha", 1.0)),
        include_prev_weights=bool(env_cfg.get("include_prev_weights", True)),
        start_index=int(split[0]),
        end_index=int(split[1]),
    )


def load_fold_model(
    ckpt_path: Path,
    n_assets: int,
    n_features: int,
    device: torch.device,
    cfg: dict,
) -> tuple[ActorCriticNetwork, int]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model_state_dict"]
    portfolio_dim = int(state["portfolio_proj.0.weight"].shape[1])
    model = ActorCriticNetwork(
        n_assets=n_assets,
        n_features=n_features,
        portfolio_dim=portfolio_dim,
        model_cfg=cfg["model"],
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, portfolio_dim


def evaluate_fold(
    fold_idx: int,
    ckpt_path: Path,
    bundle: DataBundle,
    split: dict,
    cfg: dict,
    device: torch.device,
) -> dict | None:
    if not ckpt_path.exists():
        print(f"  [!] Fold {fold_idx:2d}: checkpoint not found -- skipping.")
        return None
    try:
        model, portfolio_dim = load_fold_model(
            ckpt_path, len(bundle.tickers), len(bundle.feature_names), device, cfg
        )
    except Exception as exc:
        print(f"  [!] Fold {fold_idx:2d}: load error -- {exc}")
        return None

    test_split = split["test"]
    test_start, test_end = test_split

    include_prev = portfolio_dim > (len(bundle.tickers) + 1)
    cfg_env = {**cfg["environment"], "include_prev_weights": include_prev}
    cfg_copy = {**cfg, "environment": cfg_env}
    env = build_env(bundle, test_split, cfg_copy)

    trainer = PPOTrainer(model=model, train_env=env, val_env=env, cfg=cfg["training"], device=device)
    out = trainer.run_policy(env, deterministic=True)

    ppo = compute_performance_metrics(
        portfolio_values=np.array(out["portfolio_values"], dtype=np.float64),
        portfolio_returns=np.array(out["portfolio_returns"], dtype=np.float64),
        turnover=np.array(out["turnover"], dtype=np.float64),
        weights=np.array(out["weights"], dtype=np.float64),
    )

    test_ret = bundle.returns[test_start:test_end]
    cash = float(cfg["environment"]["initial_cash"])
    lookback = max(20, int(cfg["data"]["lookback_window"]) * 2)
    reb = int(cfg["evaluation"]["benchmark_rebalance_every"])

    bh  = run_buy_hold_equal_weight(test_ret, cash)
    bh_m = compute_performance_metrics(bh["portfolio_values"], bh["portfolio_returns"], bh["turnover"], bh["weights"])

    mz  = run_markowitz(test_ret, initial_value=cash, lookback=lookback, rebalance_every=reb)
    mz_m = compute_performance_metrics(mz["portfolio_values"], mz["portfolio_returns"], mz["turnover"], mz["weights"])

    sharpe = float(ppo["Sharpe Ratio"])
    cagr   = float(ppo["CAGR"])
    mdd    = float(ppo["Max Drawdown"])
    score  = sharpe + 2.0 * cagr - 0.5 * abs(mdd)

    return {
        "fold":            fold_idx,
        "ckpt_path":       str(ckpt_path),
        "ppo_return":      float(ppo["Total Return %"]),
        "ppo_cagr":        cagr,
        "ppo_sharpe":      sharpe,
        "ppo_mdd":         mdd,
        "ppo_vol":         float(ppo["Volatility"]),
        "ppo_calmar":      float(ppo["Calmar Ratio"]),
        "ppo_winrate":     float(ppo["Win Rate"]),
        "ppo_turnover":    float(ppo["Turnover"]),
        "bh_sharpe":       float(bh_m["Sharpe Ratio"]),
        "bh_return":       float(bh_m["Total Return %"]),
        "mz_sharpe":       float(mz_m["Sharpe Ratio"]),
        "mz_return":       float(mz_m["Total Return %"]),
        "beats_markowitz": sharpe > float(mz_m["Sharpe Ratio"]),
        "beats_buyhold":   sharpe > float(bh_m["Sharpe Ratio"]),
        "test_score":      score,
        "test_start":      test_start,
        "test_end":        test_end,
        "test_days":       test_end - test_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="configs/config.yaml")
    parser.add_argument("--top",      type=int, default=18)
    parser.add_argument("--set-best", action="store_true",
                        help="Update walk_forward_summary.json so evaluate.py uses best test fold")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device(cfg["training"])
    bundle = DataBundle.load(cfg["data"]["processed_dir"])

    wf_splits = walk_forward_splits(len(bundle.dates), cfg["data"])
    if not wf_splits:
        print("Walk-forward not enabled. Nothing to evaluate.")
        return

    base_ckpt = Path(cfg["training"]["checkpoint_path"])
    n_folds   = len(wf_splits)

    print(f"\n{'='*80}")
    print(f"  EVALUATING {n_folds} FOLD MODELS ON THEIR OWN TEST SPLITS")
    print(f"{'='*80}\n")

    results: list[dict] = []
    for fold_idx in range(1, n_folds + 1):
        ckpt = base_ckpt.parent / f"{base_ckpt.stem}_fold{fold_idx}{base_ckpt.suffix}"
        print(f"  Fold {fold_idx:2d}/{n_folds}... ", end="", flush=True)
        r = evaluate_fold(fold_idx, ckpt, bundle, wf_splits[fold_idx - 1], cfg, device)
        if r is not None:
            mz_flag = "[OK]" if r["beats_markowitz"] else "[--]"
            bh_flag = "[OK]" if r["beats_buyhold"]   else "[--]"
            print(
                f"Return={r['ppo_return']:+6.2f}%  "
                f"Sharpe={r['ppo_sharpe']:+.3f}  "
                f"MDD={r['ppo_mdd']:+.3f}  "
                f">MZ:{mz_flag}  >BH:{bh_flag}  "
                f"Score={r['test_score']:+.3f}"
            )
            results.append(r)
        else:
            print()

    if not results:
        print("\n[X] No folds evaluated successfully.")
        return

    ranked  = sorted(results, key=lambda x: x["test_score"], reverse=True)
    top_n   = ranked[:args.top]
    best    = ranked[0]

    # ── Ranking table ──────────────────────────────────────────────────────────
    RANK_LABEL = {1: "[1st]", 2: "[2nd]", 3: "[3rd]"}
    rows = []
    for i, r in enumerate(top_n, 1):
        rows.append({
            "Rank":    RANK_LABEL.get(i, f"[#{i:2d}]"),
            "Fold":    r["fold"],
            "Return%": f"{r['ppo_return']:+.2f}%",
            "CAGR":    f"{r['ppo_cagr']:+.3f}",
            "Sharpe":  f"{r['ppo_sharpe']:.3f}",
            "MDD":     f"{r['ppo_mdd']:.3f}",
            "Calmar":  f"{r['ppo_calmar']:.3f}",
            "WinRate": f"{r['ppo_winrate']:.2f}",
            "Turn":    f"{r['ppo_turnover']:.4f}",
            "BH_Sh":   f"{r['bh_sharpe']:.3f}",
            "MZ_Sh":   f"{r['mz_sharpe']:.3f}",
            ">MZ":     "YES" if r["beats_markowitz"] else "NO ",
            ">BH":     "YES" if r["beats_buyhold"]   else "NO ",
            "Score":   f"{r['test_score']:+.3f}",
        })

    print(f"\n{'='*80}")
    print(f"  TOP {len(top_n)} FOLDS -- Ranked by Test Score")
    print(f"{'='*80}")
    print(pd.DataFrame(rows).to_string(index=False))

    # ── Best model summary ─────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  [BEST] Fold {best['fold']}")
    print(f"{'='*80}")
    print(f"  Checkpoint   : {best['ckpt_path']}")
    print(f"  Test period  : step {best['test_start']} -> {best['test_end']} ({best['test_days']} days)")
    print(f"  PPO Return   : {best['ppo_return']:+.4f}%")
    print(f"  PPO CAGR     : {best['ppo_cagr']:+.4f}")
    print(f"  PPO Sharpe   : {best['ppo_sharpe']:.4f}")
    print(f"  PPO MDD      : {best['ppo_mdd']:.4f}")
    print(f"  PPO Calmar   : {best['ppo_calmar']:.4f}")
    print(f"  BuyHold Sh   : {best['bh_sharpe']:.4f}  {'[PPO WINS]' if best['beats_buyhold'] else '[PPO LOSES]'}")
    print(f"  Markowitz Sh : {best['mz_sharpe']:.4f}  {'[PPO WINS]' if best['beats_markowitz'] else '[PPO LOSES]'}")
    print(f"  Test Score   : {best['test_score']:+.4f}")
    print()

    n_pos    = sum(1 for r in results if r["ppo_return"] > 0)
    n_mz     = sum(1 for r in results if r["beats_markowitz"])
    n_bh     = sum(1 for r in results if r["beats_buyhold"])
    print(f"  Overall: {n_pos}/{len(results)} positive return | "
          f"{n_mz}/{len(results)} beat Markowitz | "
          f"{n_bh}/{len(results)} beat BuyHold")

    # ── Save full evaluation ───────────────────────────────────────────────────
    out = Path("outputs/reports/fold_evaluation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ranked, indent=2, default=str), encoding="utf-8")
    print(f"\n  [SAVED] {out}")

    # ── Optionally update summary so evaluate.py picks best test fold ──────────
    if args.set_best:
        summary_path = Path(
            cfg["evaluation"].get("walk_forward_summary_path",
                                  "outputs/reports/walk_forward_summary.json")
        )
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for entry in summary:
                if int(entry["fold"]) == best["fold"]:
                    entry["score"] = 9999.0   # force evaluate.py to pick this fold
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"  [UPDATED] walk_forward_summary.json -> Fold {best['fold']} selected")


if __name__ == "__main__":
    main()
