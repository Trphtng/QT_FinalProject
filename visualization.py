from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


US_CONFIG = "configs/config.yaml"
VN_CONFIG = "configs/config_vn30.yaml"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_samples_from_raw_dir(config_path: str) -> dict:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["data"]["raw_dir"])

    total_rows = 0
    n_files = 0
    tickers = []

    for csv_file in raw_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        if df.empty:
            continue

        n_files += 1
        total_rows += len(df)
        tickers.append(csv_file.stem)

    return {
        "dataset_name": cfg["project"]["name"],
        "num_samples": total_rows,
        "num_assets": n_files,
        "tickers": tickers,
    }


def plot_sample_counts(us_info: dict, vn_info: dict) -> None:
    labels = ["US Dataset", "VN Dataset"]
    sample_counts = [us_info["num_samples"], vn_info["num_samples"]]
    asset_counts = [us_info["num_assets"], vn_info["num_assets"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = axes[0].bar(labels, sample_counts, color=["#1f77b4", "#d62728"], width=0.6)
    axes[0].set_title("Number of Samples")
    axes[0].set_ylabel("Rows")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom")

    bars2 = axes[1].bar(labels, asset_counts, color=["#1f77b4", "#d62728"], width=0.6)
    axes[1].set_title("Number of Assets")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom")

    fig.suptitle("Dataset", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    us_info = count_samples_from_raw_dir(US_CONFIG)
    vn_info = count_samples_from_raw_dir(VN_CONFIG)
    plot_sample_counts(us_info, vn_info)
