"""Run train/evaluate for multiple seeds and summarize metrics."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


CONFIG_PATH = Path("configs/config.yaml")
OUTPUT_DIR = Path("outputs/reports")


def _run_command(args: list[str]) -> None:
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}")


def main() -> None:
    seeds = [42, 52, 62]
    original_text = CONFIG_PATH.read_text(encoding="utf-8")
    original_cfg = yaml.safe_load(original_text)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, float | int | str]] = []

    try:
        for seed in seeds:
            cfg = copy.deepcopy(original_cfg)
            cfg["project"]["seed"] = int(seed)
            cfg["training"]["resume_from"] = None
            cfg["training"]["checkpoint_path"] = f"outputs/models/best_model_seed_{seed}.pt"
            cfg["training"]["tensorboard_dir"] = f"outputs/tensorboard/seed_{seed}"
            cfg["evaluation"]["report_path"] = f"outputs/reports/metrics_summary_seed_{seed}.json"
            cfg["evaluation"]["comparison_csv"] = f"outputs/reports/baseline_comparison_seed_{seed}.csv"
            CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            print(f"[multi-seed] Running seed={seed}: train.py")
            _run_command([sys.executable, "train.py"])
            print(f"[multi-seed] Running seed={seed}: evaluate.py")
            _run_command([sys.executable, "evaluate.py"])

            report_path = Path(cfg["evaluation"]["report_path"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            ppo = report["PPO Actor-Critic"]
            row = {
                "seed": seed,
                "Total Return %": float(ppo["Total Return %"]),
                "CAGR": float(ppo["CAGR"]),
                "Sharpe Ratio": float(ppo["Sharpe Ratio"]),
                "Sortino Ratio": float(ppo["Sortino Ratio"]),
                "Max Drawdown": float(ppo["Max Drawdown"]),
                "Volatility": float(ppo["Volatility"]),
                "Calmar Ratio": float(ppo["Calmar Ratio"]),
                "Win Rate": float(ppo["Win Rate"]),
                "Final Portfolio Value": float(ppo["Final Portfolio Value"]),
                "Turnover": float(ppo["Turnover"]),
                "Avg Holding Period": float(ppo["Avg Holding Period"]),
            }
            per_seed_rows.append(row)
    finally:
        CONFIG_PATH.write_text(original_text, encoding="utf-8")

    per_seed_df = pd.DataFrame(per_seed_rows)
    summary_df = per_seed_df.drop(columns=["seed"]).agg(["mean", "std"]).T.reset_index()
    summary_df.columns = ["Metric", "Mean", "Std"]

    per_seed_path = OUTPUT_DIR / "ppo_multiseed_per_seed.csv"
    summary_path = OUTPUT_DIR / "ppo_multiseed_summary.csv"
    per_seed_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[multi-seed] Saved per-seed metrics: {per_seed_path}")
    print(f"[multi-seed] Saved summary mean/std: {summary_path}")
    print(per_seed_df.to_string(index=False))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
